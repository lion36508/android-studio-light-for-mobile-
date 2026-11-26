from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .export import _quantized_parameter_names
from .generate import load_addsub_model
from .quantization import WEIGHT_SYMBOLS, quantize_to_codes, weight_scale
from .train import sha256_file


def audit_model_quantization(model: torch.nn.Module) -> dict[str, Any]:
    """Inspect the exact model tensors selected for packed five-symbol export."""
    parameters = dict(model.named_parameters())
    quantized_names = sorted(_quantized_parameter_names(model))
    missing = sorted(set(quantized_names) - set(parameters))
    if missing:
        raise ValueError(f"Quantized parameter names are missing from the model: {missing}")
    if not quantized_names:
        raise ValueError("No quantized model parameters were found.")

    fixed_scale_names = [
        name for name in quantized_names
        if getattr(parameters[name], "_qmini_fixed_scale", None) is not None
    ]
    if fixed_scale_names and len(fixed_scale_names) != len(quantized_names):
        raise ValueError("A model cannot mix legacy fixed scales and dynamic per-tensor scales.")
    scale_mode = "legacy-fixed" if fixed_scale_names else "dynamic-per-tensor"

    tensor_reports = []
    total_histogram = np.zeros(len(WEIGHT_SYMBOLS), dtype=np.int64)
    total_elements = 0

    for name in quantized_names:
        parameter = parameters[name]
        if not parameter.is_floating_point():
            raise TypeError(f"Quantized parameter {name!r} is not floating point.")
        if parameter.numel() < 1 or not bool(torch.isfinite(parameter).all()):
            raise ValueError(f"Quantized parameter {name!r} is empty or non-finite.")

        scale = float(weight_scale(parameter).detach().cpu().item())
        values = parameter.detach().cpu().numpy()
        codes = quantize_to_codes(values, scale=scale)
        histogram = np.bincount(codes.reshape(-1), minlength=len(WEIGHT_SYMBOLS))
        if int(histogram.sum()) != int(parameter.numel()):
            raise RuntimeError(f"Symbol histogram does not cover every value in {name!r}.")

        count = int(histogram.sum())
        total_histogram += histogram
        total_elements += count
        tensor_reports.append({
            "name": name,
            "shape": list(parameter.shape),
            "element_count": count,
            "scale": scale,
            "master_weight_min": float(values.min()),
            "master_weight_max": float(values.max()),
            "master_weight_mean_abs": float(np.abs(values).mean()),
            "symbols": [
                {
                    "code": index,
                    "value": float(symbol),
                    "count": int(histogram[index]),
                    "fraction": float(histogram[index] / count),
                }
                for index, symbol in enumerate(WEIGHT_SYMBOLS)
            ],
            "zero_symbol_fraction": float(histogram[2] / count),
            "used_symbol_count": int(np.count_nonzero(histogram)),
        })

    return {
        "format": "qmini-addsub-quantization-audit-v1",
        "weight_format": (
            "legacy-fixed-scale-five-symbol-q3"
            if scale_mode == "legacy-fixed"
            else "scaled-five-symbol-q3"
        ),
        "quantization_scale_mode": scale_mode,
        "bits_per_weight": 3,
        "symbol_values": [float(value) for value in WEIGHT_SYMBOLS],
        "quantized_parameter_count": len(quantized_names),
        "quantized_element_count": total_elements,
        "overall_symbols": [
            {
                "code": index,
                "value": float(symbol),
                "count": int(total_histogram[index]),
                "fraction": float(total_histogram[index] / total_elements),
            }
            for index, symbol in enumerate(WEIGHT_SYMBOLS)
        ],
        "overall_zero_symbol_fraction": float(total_histogram[2] / total_elements),
        "overall_used_symbol_count": int(np.count_nonzero(total_histogram)),
        "tensors": tensor_reports,
        "interpretation_limit": (
            "This report measures the simulated five-symbol quantization of the "
            "checkpoint's master weights using the export quantizer and scales. "
            "It does not prove integer-only inference, hardware execution, model "
            "quality, or that every symbol must be used equally."
        ),
    }


def audit_addsub_checkpoint(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device_name: str = "cpu",
) -> dict[str, Any]:
    checkpoint_file = Path(checkpoint_path)
    tokenizer_file = Path(tokenizer_path)
    if not checkpoint_file.is_file() or not tokenizer_file.is_file():
        raise FileNotFoundError("Checkpoint and tokenizer files are both required.")

    metadata = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    run_contract = metadata.get("run_contract", {})
    if isinstance(run_contract, dict) and run_contract.get("wavebit_hsra") is not None:
        raise ValueError(
            "WaveBit-HSRA checkpoints use a separate quantization path and are not "
            "covered by this five-symbol codebook audit."
        )

    model = load_addsub_model(
        checkpoint_file,
        device_name=device_name,
        tokenizer_path=tokenizer_file,
    )
    if not hasattr(model, "config") or not hasattr(model.config, "to_dict"):
        raise RuntimeError("Loaded checkpoint model has no serializable Add/Sub config.")

    report = audit_model_quantization(model)
    report.update({
        "checkpoint_sha256": sha256_file(checkpoint_file),
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "checkpoint_format": metadata.get("format"),
        "checkpoint_step": int(metadata.get("step", 0)),
        "model_config": model.config.to_dict(),
        "device": device_name,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit five-symbol Add/Sub quantization on a real checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = audit_addsub_checkpoint(
        args.checkpoint,
        args.tokenizer,
        device_name=args.device,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError("Quantization report already exists; choose a new output path.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"Temporary report path already exists: {temporary}")
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    print(serialized, end="")


if __name__ == "__main__":
    main()
