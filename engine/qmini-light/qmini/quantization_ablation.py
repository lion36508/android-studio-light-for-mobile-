from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qmini.config import QMiniConfig
from qmini.quantization import (
    pack_int4_groupwise,
    pack_quaternary,
    pack_ternary,
    quantization_error_metrics,
    unpack_int4_groupwise,
    unpack_quaternary,
    unpack_ternary,
    quantize_int8_rowwise,
    dequantize_int8_rowwise,
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_accumulator() -> dict[str, float]:
    return {"reference_sq": 0.0, "error_sq": 0.0, "dot": 0.0, "approx_sq": 0.0, "max_abs": 0.0}


def _add_metrics(total: dict[str, float], reference: torch.Tensor, approximation: torch.Tensor) -> None:
    ref = reference.detach().float().reshape(-1)
    approx = approximation.detach().float().reshape(-1)
    difference = approx - ref
    total["reference_sq"] += float(torch.sum(ref * ref))
    total["error_sq"] += float(torch.sum(difference * difference))
    total["dot"] += float(torch.sum(ref * approx))
    total["approx_sq"] += float(torch.sum(approx * approx))
    total["max_abs"] = max(total["max_abs"], float(difference.abs().max()))


def _finish_metrics(total: dict[str, float]) -> dict[str, float]:
    relative_l2 = math.sqrt(total["error_sq"]) / max(math.sqrt(total["reference_sq"]), 1e-12)
    cosine = total["dot"] / max(
        math.sqrt(total["reference_sq"] * total["approx_sq"]), 1e-12
    )
    return {
        "relative_l2_error": relative_l2,
        "cosine_similarity": max(-1.0, min(1.0, cosine)),
        "max_absolute_error": total["max_abs"],
    }


def _weight_component(name: str) -> str:
    key = name.lower()
    if "embedding" in key or "token_embed" in key:
        return "embedding"
    if "router" in key:
        return "router"
    if "expert" in key or ".experts." in key:
        return "expert"
    if any(part in key for part in ("depthwise", ".conv.", "convolution")):
        return "convolution"
    if any(part in key for part in ("state_space", ".ssm.", "state_mixer")):
        return "state_space"
    if "lm_head" in key or "output_projection" in key:
        return "output_projection"
    return "other"


def _scale_summary(scales: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(scales, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "count": 0, "zero_count": 0, "min": 0.0, "median": 0.0,
            "p95": 0.0, "max": 0.0, "mean": 0.0,
        }
    return {
        "count": int(values.size),
        "zero_count": int(np.count_nonzero(values == 0.0)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def run_ablation(
    model_dir: Path,
    output_path: Path,
    group_size: int,
    checkpoint_path: Path | None = None,
    eval_tokens: Path | None = None,
    eval_mask: Path | None = None,
    eval_batch_size: int = 2,
    eval_seq_len: int = 128,
    eval_batches: int = 100,
    eval_seed: int = 42,
    eval_device: str = "cpu",
    max_eval_parameters: int = 10_000_000,
) -> dict[str, Any]:
    """Compare formats using a Q-Mini final snapshot or own training checkpoint."""
    source_path = Path(model_dir).expanduser()
    if checkpoint_path is not None:
        weight_path = Path(checkpoint_path).expanduser()
    elif source_path.is_file():
        weight_path = source_path
    elif (source_path / "model.pt").is_file():
        weight_path = source_path / "model.pt"
    elif (source_path / "latest.pt").is_file():
        weight_path = source_path / "latest.pt"
    else:
        raise FileNotFoundError(
            "Use a legacy Q-Mini final_model with its verified model.pt, a training "
            "run directory containing latest.pt, or an explicit --checkpoint. New Q2-only "
            "deployment archives are not unquantized references. External/pretrained weights "
            "are not accepted."
        )

    loaded = torch.load(weight_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError("Selected Q-Mini weight source is not a non-empty dictionary.")

    if loaded.get("format") in {"qmini-checkpoint-v1", "qmini-checkpoint-v2"}:
        if not isinstance(loaded.get("model"), dict) or not isinstance(loaded.get("config"), dict):
            raise ValueError("Q-Mini training checkpoint is missing its model/config state.")
        state = loaded["model"]
        config = QMiniConfig.from_dict(loaded["config"])
        source_kind = "qmini-training-checkpoint"
    else:
        config_path = (
            source_path / "config.json"
            if source_path.is_dir()
            else weight_path.parent / "config.json"
        )
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Config file not found beside model weights: {config_path}"
            )
        config = QMiniConfig.from_json(config_path)
        manifest_path = config_path.parent / "model_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "A Q-Mini final_model/model.pt source must include its "
                "model_manifest.json provenance record."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "qmini-final-model-v2":
            raise ValueError("Unsupported or missing Q-Mini final model manifest format.")
        try:
            manifest_config = QMiniConfig.from_dict(manifest.get("config"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Final model manifest contains an invalid config.") from exc
        if manifest_config.to_dict() != config.to_dict():
            raise ValueError("Final model manifest config does not match config.json.")
        if int(manifest.get("parameters", -1)) != config.estimated_parameter_count():
            raise ValueError("Final model manifest parameter count does not match config.json.")
        packed_name = manifest.get("packed_export")
        packed_bytes = manifest.get("packed_export_bytes")
        if (
            not isinstance(packed_name, str)
            or not packed_name
            or Path(packed_name).name != packed_name
            or not packed_name.endswith(".npz")
        ):
            raise ValueError(
                "Final model manifest must identify a local .npz packed Q2 export."
            )
        packed_path = config_path.parent / packed_name
        if not packed_path.is_file() or packed_path.stat().st_size != packed_bytes:
            raise ValueError("Final model manifest packed-export size/path fingerprint is invalid.")
        if manifest.get("packed_export_sha256") != _sha256_path(packed_path):
            raise ValueError("Final model manifest packed-export SHA-256 does not match the archive.")
        dense_snapshot_bytes = manifest.get("dense_snapshot_bytes")
        if dense_snapshot_bytes != weight_path.stat().st_size:
            raise ValueError("Final model manifest dense snapshot byte count does not match model.pt.")
        if manifest.get("dense_snapshot_sha256") != _sha256_path(weight_path):
            raise ValueError("Final model manifest dense snapshot SHA-256 does not match model.pt.")
        state = loaded
        source_kind = "qmini-final-model-state-dict"

    if not isinstance(state, dict) or not state:
        raise ValueError("Q-Mini model state is not a non-empty tensor dictionary.")

    formats = ("fp16", "int8", "q2", "ternary", "int4")
    totals = {name: _metric_accumulator() for name in formats}
    byte_counts = {name: 0 for name in formats}
    per_tensor: dict[str, Any] = {}
    payload_breakdown = {
        name: {
            "matrix_symbol_or_fp16_bytes": 0,
            "scale_bytes": 0,
            "vector_bytes": 0,
        }
        for name in formats
    }
    components = (
        "embedding", "output_projection", "router", "expert",
        "convolution", "state_space", "other",
    )
    component_totals = {
        component: {name: _metric_accumulator() for name in formats}
        for component in components
    }
    component_counts = {component: 0 for component in components}
    q2_scale_reports: dict[str, dict[str, float | int]] = {}
    quality_states: dict[str, dict[str, torch.Tensor]] | None = (
        {name: {} for name in formats} if eval_tokens is not None else None
    )
    matrix_count = 0
    vector_count = 0

    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            continue
        reference = value.detach().float().cpu()
        if reference.ndim < 2:
            vector_count += 1
            approximation = reference.half().float()
            _add_metrics(totals["fp16"], reference, approximation)
            per_tensor[name] = {
                "shape": list(reference.shape),
                "kind": "fp16-vector",
                "fp16": quantization_error_metrics(reference, approximation),
            }
            for format_name in ("int8", "q2", "ternary", "int4"):
                _add_metrics(totals[format_name], reference, approximation)
            vector_bytes = reference.numel() * 2
            component = _weight_component(name)
            component_counts[component] += 1
            for format_name in formats:
                byte_counts[format_name] += vector_bytes
                payload_breakdown[format_name]["vector_bytes"] += vector_bytes
                _add_metrics(component_totals[component][format_name], reference, approximation)
                if quality_states is not None:
                    quality_states[format_name][name] = approximation.reshape(value.shape).float()
            continue

        matrix_count += 1
        component = _weight_component(name)
        component_counts[component] += 1
        flat = reference.reshape(reference.shape[0], -1)
        row_report: dict[str, Any] = {
            "component": component,
            "shape": list(reference.shape),
            "flattened_matrix_shape": list(flat.shape),
            "kind": "quantized-matrix",
            "formats": {},
        }

        fp16 = flat.half().float()
        _add_metrics(totals["fp16"], flat, fp16)
        byte_counts["fp16"] += flat.numel() * 2
        payload_breakdown["fp16"]["matrix_symbol_or_fp16_bytes"] += flat.numel() * 2
        _add_metrics(component_totals[component]["fp16"], flat, fp16)
        row_report["formats"]["fp16"] = {
            **quantization_error_metrics(flat, fp16),
            "bytes": flat.numel() * 2,
        }

        int8_values, int8_scales = quantize_int8_rowwise(flat)
        int8 = dequantize_int8_rowwise(int8_values, int8_scales)
        int8_bytes = int(int8_values.numel() + int8_scales.numel() * 4)
        _add_metrics(totals["int8"], flat, int8)
        byte_counts["int8"] += int8_bytes
        payload_breakdown["int8"]["matrix_symbol_or_fp16_bytes"] += int8_values.numel()
        payload_breakdown["int8"]["scale_bytes"] += int8_scales.numel() * 4
        _add_metrics(component_totals[component]["int8"], flat, int8)
        row_report["formats"]["int8"] = {
            **quantization_error_metrics(flat, int8),
            "bytes": int8_bytes,
        }

        q2_packed, q2_scales = pack_quaternary(flat)
        q2 = torch.from_numpy(
            unpack_quaternary(q2_packed, q2_scales, flat.shape[1])
        )
        q2_bytes = int(q2_packed.nbytes + q2_scales.nbytes)
        _add_metrics(totals["q2"], flat, q2)
        byte_counts["q2"] += q2_bytes
        payload_breakdown["q2"]["matrix_symbol_or_fp16_bytes"] += q2_packed.nbytes
        payload_breakdown["q2"]["scale_bytes"] += q2_scales.nbytes
        _add_metrics(component_totals[component]["q2"], flat, q2)
        q2_scale_reports[name] = _scale_summary(q2_scales)
        row_report["formats"]["q2"] = {
            **quantization_error_metrics(flat, q2),
            "bytes": q2_bytes,
            "packed_symbol_bytes": int(q2_packed.nbytes),
            "row_scale_bytes": int(q2_scales.nbytes),
            "row_scale_summary": q2_scale_reports[name],
        }

        ternary_packed, ternary_scales = pack_ternary(flat)
        ternary = torch.from_numpy(
            unpack_ternary(ternary_packed, ternary_scales, flat.shape[1])
        )
        ternary_bytes = int(ternary_packed.nbytes + ternary_scales.nbytes)
        _add_metrics(totals["ternary"], flat, ternary)
        byte_counts["ternary"] += ternary_bytes
        payload_breakdown["ternary"]["matrix_symbol_or_fp16_bytes"] += ternary_packed.nbytes
        payload_breakdown["ternary"]["scale_bytes"] += ternary_scales.nbytes
        _add_metrics(component_totals[component]["ternary"], flat, ternary)
        row_report["formats"]["ternary"] = {
            **quantization_error_metrics(flat, ternary),
            "bytes": ternary_bytes,
        }

        int4_packed, int4_scales = pack_int4_groupwise(flat, group_size=group_size)
        int4 = torch.from_numpy(
            unpack_int4_groupwise(
                int4_packed, int4_scales, flat.shape[1], group_size=group_size
            )
        )
        int4_bytes = int(int4_packed.nbytes + int4_scales.nbytes)
        _add_metrics(totals["int4"], flat, int4)
        byte_counts["int4"] += int4_bytes
        payload_breakdown["int4"]["matrix_symbol_or_fp16_bytes"] += int4_packed.nbytes
        payload_breakdown["int4"]["scale_bytes"] += int4_scales.nbytes
        _add_metrics(component_totals[component]["int4"], flat, int4)
        row_report["formats"]["int4"] = {
            **quantization_error_metrics(flat, int4),
            "bytes": int4_bytes,
            "group_size": group_size,
        }
        per_tensor[name] = row_report
        if quality_states is not None:
            quality_states["fp16"][name] = fp16.reshape(value.shape).float()
            quality_states["int8"][name] = int8.reshape(value.shape).float()
            quality_states["q2"][name] = q2.reshape(value.shape).float()
            quality_states["ternary"][name] = ternary.reshape(value.shape).float()
            quality_states["int4"][name] = int4.reshape(value.shape).float()

    if matrix_count == 0:
        raise ValueError("No matrix weights were found in this model state dictionary.")

    heldout_evaluation = None
    if eval_tokens is not None:
        if isinstance(max_eval_parameters, bool) or max_eval_parameters < 1:
            raise ValueError("max_eval_parameters must be a positive integer.")
        if config.estimated_parameter_count() > max_eval_parameters:
            raise ValueError(
                f"Held-out quantization evaluation is capped at {max_eval_parameters:,} "
                f"parameters to bound the five candidate state copies; this model has "
                f"{config.estimated_parameter_count():,}. Raise --max-eval-parameters "
                "only when sufficient host RAM is available."
            )
        if eval_batch_size < 1 or eval_seq_len < 2 or eval_batches < 1:
            raise ValueError("Evaluation batch size, sequence length and batch count must be valid.")
        if eval_seq_len > config.max_context:
            raise ValueError("Evaluation sequence length exceeds the model's max_context.")
        if eval_mask is not None and not Path(eval_mask).is_file():
            raise FileNotFoundError(f"Evaluation mask not found: {eval_mask}")
        if not Path(eval_tokens).is_file():
            raise FileNotFoundError(f"Evaluation token stream not found: {eval_tokens}")

        # Keep INT8 activation fake quantization fixed across candidates. Only
        # the matrix weights differ; model's Q2 STE must not re-quantize them.
        import qmini.model as model_module
        from qmini.model import QMiniMoE
        from qmini.evaluate import evaluate_language_model

        original_q2_fake_quantizer = model_module.quantize_quaternary
        model_module.quantize_quaternary = lambda weight: weight
        heldout_evaluation = {}
        try:
            for format_name in formats:
                candidate = QMiniMoE(config)
                candidate.load_state_dict(quality_states[format_name], strict=True)
                candidate.to(torch.device(eval_device)).eval()
                result = evaluate_language_model(
                    candidate,
                    eval_tokens,
                    mask_path=eval_mask,
                    batch_size=eval_batch_size,
                    seq_len=eval_seq_len,
                    batches=eval_batches,
                    seed=eval_seed,
                )
                heldout_evaluation[format_name] = {
                    "cross_entropy": result["cross_entropy"],
                    "perplexity": result["perplexity"],
                    "supervised_target_tokens": result["supervised_target_tokens"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "input_tokens_per_second": result["input_tokens_per_second"],
                    "seed": eval_seed,
                    "batch_size": eval_batch_size,
                    "sequence_length": eval_seq_len,
                    "batches": eval_batches,
                    "device": str(eval_device),
                    "activation_path": "INT8 fake quantization held constant",
                }
                del candidate
                if str(eval_device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            model_module.quantize_quaternary = original_q2_fake_quantizer

    category_metrics = {
        component: {
            "tensor_count": component_counts[component],
            "formats": {
                format_name: _finish_metrics(component_totals[component][format_name])
                for format_name in formats
            },
        }
        for component in components
    }
    for format_name in formats:
        payload_breakdown[format_name]["total_weight_payload_bytes"] = byte_counts[format_name]

    packed_archive_candidates = []
    if source_path.is_dir():
        packed_archive_candidates.extend([
            source_path / "model.qmini.npz",
            source_path / "final_model" / "model.qmini.npz",
        ])
    elif source_path.is_file():
        packed_archive_candidates.append(source_path.parent / "model.qmini.npz")
    packed_archive = next(
        (candidate for candidate in packed_archive_candidates if candidate.is_file()),
        None,
    )

    report = {
        "format": "qmini-weight-quantization-ablation-v3",
        "source_model_dir": str(source_path.resolve()),
        "source_weight_path": str(weight_path.resolve()),
        "source_kind": source_kind,
        "source_config": config.to_dict(),
        "source_parameter_count": config.estimated_parameter_count(),
        "matrix_tensor_count": matrix_count,
        "vector_tensor_count": vector_count,
        "component_metrics": category_metrics,
        "q2_row_scale_reports": q2_scale_reports,
        "payload_breakdown": payload_breakdown,
        "packed_archive": (
            {
                "path": str(packed_archive.resolve()),
                "file_bytes": int(packed_archive.stat().st_size),
            }
            if packed_archive is not None else None
        ),
        "group_size": group_size,
        "heldout_language_evaluation": heldout_evaluation,
        "evaluation_dataset_sha256": (
            _sha256_path(Path(eval_tokens))
            if eval_tokens is not None else None
        ),
        "evaluation_mask_sha256": (
            _sha256_path(Path(eval_mask))
            if eval_mask is not None else None
        ),
        "formats": {
            name: {
                **_finish_metrics(totals[name]),
                "estimated_weight_payload_bytes": byte_counts[name],
            }
            for name in formats
        },
        "tensor_results": per_tensor,
        "limitations": [
            "Weight reconstruction metrics are always reported; held-out NLL/perplexity are reported only when --eval-tokens is supplied.",
            "Reconstruction metrics and held-out language-model loss are not semantic assistant-quality scores; task-level capability must be evaluated separately.",
            "All formats are experimental; this script does not change the model's default quantizer.",
            "Only Q-Mini-owned final_model/model.pt or its own latest.pt training checkpoint is accepted; no external pretrained weights are loaded.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare FP16, rowwise INT8, Q2, zero-preserving ternary and groupwise INT4 weight error."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir", help="legacy final_model with model.pt or training run directory with latest.pt")
    source.add_argument("--checkpoint", help="Q-Mini-owned latest.pt training checkpoint")
    parser.add_argument("--output", default="quantization-ablation.json")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--eval-tokens", default=None, help="optional held-out uint16 token stream for NLL/perplexity")
    parser.add_argument("--eval-mask", default=None, help="optional aligned uint8 loss mask")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-seq-len", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--max-eval-parameters", type=int, default=10_000_000)
    args = parser.parse_args()
    if args.group_size < 1:
        parser.error("--group-size must be positive.")
    model_source = (
        Path(args.model_dir).expanduser()
        if args.model_dir is not None
        else Path(args.checkpoint).expanduser()
    )
    report = run_ablation(
        model_source,
        Path(args.output).expanduser(),
        args.group_size,
        eval_tokens=Path(args.eval_tokens).expanduser() if args.eval_tokens else None,
        eval_mask=Path(args.eval_mask).expanduser() if args.eval_mask else None,
        eval_batch_size=args.eval_batch_size,
        eval_seq_len=args.eval_seq_len,
        eval_batches=args.eval_batches,
        eval_seed=args.eval_seed,
        eval_device=args.eval_device,
        max_eval_parameters=args.max_eval_parameters,
    )
    for name, result in report["formats"].items():
        print(
            f"{name}: rel_l2={result['relative_l2_error']:.6f} "
            f"cosine={result['cosine_similarity']:.6f} "
            f"bytes={result['estimated_weight_payload_bytes']:,}"
        )
    if report.get("heldout_language_evaluation"):
        for name, result in report["heldout_language_evaluation"].items():
            print(
                f"{name} heldout_nll={result['cross_entropy']:.6f} "
                f"ppl={result['perplexity']}"
            )
    print(f"Report saved: {Path(args.output).expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
