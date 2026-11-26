from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch

from qmini.config import QMiniConfig
from qmini.model import QMiniMoE
from qmini.native import native_kernel_status


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_process_rss_bytes() -> int | None:
    """Return process peak RSS where the platform exposes getrusage."""
    try:
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    # macOS reports bytes; Linux and the supported BSD CI targets report KiB.
    return value if platform.system() == "Darwin" else value * 1024


def cpu_model_name() -> str:
    name = platform.processor().strip()
    if name and not name.isdigit():
        return name
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                lowered = line.lower()
                if lowered.startswith(("model name", "hardware", "cpu model")):
                    candidate = line.split(":", 1)[-1].strip()
                    if candidate and not candidate.isdigit():
                        return candidate
        except OSError:
            pass
    return platform.machine() or "unknown"


@torch.inference_mode()
def prefill(
    model: QMiniMoE,
    prompt_ids: torch.Tensor,
    thinking_steps: int,
):
    """Build the incremental convolution caches for a batched prompt."""
    state = None
    output = None
    for position in range(prompt_ids.shape[1]):
        output, state = model.forward_step(
            prompt_ids[:, position],
            state=state,
            thinking_steps=thinking_steps,
        )
    if state is None or output is None:
        raise RuntimeError("Benchmark prefill did not create a model state.")
    return output, state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Q-Mini full forward, prompt prefill and cached decode throughput."
    )
    parser.add_argument("--config", default="configs/qmini_light_0.7m.json")
    parser.add_argument(
        "--model-dir", default=None,
        help="load a trained model bundle; prefers the packed Q2 CPU runtime when available",
    )
    parser.add_argument(
        "--device", default=None,
        help="execution device; leave unset to auto-select (packed model bundles default to CPU)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128, help="Synthetic prompt length.")
    parser.add_argument("--iterations", type=int, default=20, help="Measured iterations or decode tokens.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--thinking-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("forward", "decode", "both"), default="both")
    parser.add_argument("--output", default=None, help="optional JSON path for this measured report")
    args = parser.parse_args()

    if args.batch_size < 1 or args.iterations < 1 or args.warmup < 0:
        raise SystemExit("batch-size and iterations must be positive; warmup cannot be negative.")

    if args.seed < 0:
        raise SystemExit("seed must be non-negative.")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.model_dir:
        from qmini.generate import load_model
        model, _tokenizer, device = load_model(args.model_dir, args.device)
        config = model.config
    else:
        config = QMiniConfig.from_json(args.config)
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = QMiniMoE(config).to(device).eval()

    if not 1 <= args.seq_len <= config.max_context:
        raise SystemExit("seq-len must be within the loaded model's configured context.")
    if not 1 <= args.thinking_steps <= config.max_thinking_steps:
        raise SystemExit(
            f"thinking-steps must be 1..{config.max_thinking_steps} for this model."
        )
    prompt_ids = torch.randint(
        0, config.vocab_size, (args.batch_size, args.seq_len), device=device
    )

    backend = native_kernel_status()
    report = {
        "format": "qmini-benchmark-v1",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cpu_model": cpu_model_name(),
        "runtime_model": (
            "trained-packed-q2"
            if bool(getattr(model, "_qmini_packed_runtime", False))
            else "trained-dense"
            if args.model_dir
            else "random-initialized-config"
        ),
        "packed_q2_runtime": bool(getattr(model, "_qmini_packed_runtime", False)),
        "cpu_projection_backend": backend["backend"],
        "arm_neon": backend["uses_neon"],
        "parameters": model.parameter_report()["total"],
        "config": config.to_dict(),
        "batch_size": args.batch_size,
        "prompt_tokens": args.batch_size * args.seq_len,
        "thinking_steps": args.thinking_steps,
        "seed": args.seed,
        "warmup": args.warmup,
        "measured_iterations": args.iterations,
        "mode": args.mode,
        "input_note": "synthetic token IDs; tokenizer and sampling overhead excluded",
    }
    print(f"python={report['python']}")
    print(f"torch={report['pytorch']}")
    print(f"platform={report['platform']}")
    print(f"device={report['device']}")
    print(f"cpu_model={report['cpu_model']}")
    print(f"runtime_model={report['runtime_model']}")
    print(f"packed_q2_runtime={report['packed_q2_runtime']}")
    print(f"cpu_projection_backend={report['cpu_projection_backend']}")
    print(f"arm_neon={report['arm_neon']}")
    print(f"parameters={report['parameters']:,}")
    print(f"prompt_tokens={report['prompt_tokens']}")
    print(f"thinking_steps={report['thinking_steps']}")
    print(f"warmup={report['warmup']} measured_iterations={report['measured_iterations']}")
    print(report["input_note"])

    with torch.inference_mode():
        if args.mode in {"forward", "both"}:
            for _ in range(args.warmup):
                model(prompt_ids, thinking_steps=args.thinking_steps)
            synchronize(device)

            start = time.perf_counter()
            for _ in range(args.iterations):
                model(prompt_ids, thinking_steps=args.thinking_steps)
            synchronize(device)
            elapsed = max(time.perf_counter() - start, 1e-9)
            tokens = args.batch_size * args.seq_len * args.iterations
            report["forward_elapsed_seconds"] = elapsed
            report["forward_input_tokens_per_second"] = tokens / elapsed
            print(f"forward_elapsed_seconds={elapsed:.4f}")
            print(f"forward_input_tokens_per_second={tokens / elapsed:.2f}")

        if args.mode in {"decode", "both"}:
            # Warm up the same streaming path used by actual autoregressive generation.
            _, warm_state = prefill(model, prompt_ids, args.thinking_steps)
            for _ in range(args.warmup):
                next_ids = torch.randint(
                    0, config.vocab_size, (args.batch_size,), device=device
                )
                _, warm_state = model.forward_step(next_ids, state=warm_state)
            synchronize(device)

            prefill_start = time.perf_counter()
            _, state = prefill(model, prompt_ids, args.thinking_steps)
            synchronize(device)
            prefill_elapsed = max(time.perf_counter() - prefill_start, 1e-9)

            decode_start = time.perf_counter()
            for _ in range(args.iterations):
                next_ids = torch.randint(
                    0, config.vocab_size, (args.batch_size,), device=device
                )
                _, state = model.forward_step(next_ids, state=state)
            synchronize(device)
            decode_elapsed = max(time.perf_counter() - decode_start, 1e-9)
            decoded_tokens = args.batch_size * args.iterations

            report["prefill_elapsed_seconds"] = prefill_elapsed
            report["prefill_input_tokens_per_second"] = (
                args.batch_size * args.seq_len / prefill_elapsed
            )
            report["decode_elapsed_seconds"] = decode_elapsed
            report["decode_tokens_per_second"] = decoded_tokens / decode_elapsed
            report["decode_tokens_per_request"] = args.iterations
            print(f"prefill_elapsed_seconds={prefill_elapsed:.4f}")
            print(
                f"prefill_input_tokens_per_second="
                f"{report['prefill_input_tokens_per_second']:.2f}"
            )
            print(f"decode_elapsed_seconds={decode_elapsed:.4f}")
            print(f"decode_tokens_per_second={report['decode_tokens_per_second']:.2f}")
            print(f"decode_tokens_per_request={args.iterations}")

        if device.type == "cuda":
            report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
            print(f"peak_cuda_memory_bytes={report['peak_cuda_memory_bytes']}")
        rss = peak_process_rss_bytes()
        report["peak_process_rss_bytes"] = rss
        print(f"peak_process_rss_bytes={rss if rss is not None else 'unavailable'}")

    if args.output:
        destination = Path(args.output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, default=list, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"benchmark_report={destination}")


if __name__ == "__main__":
    main()
