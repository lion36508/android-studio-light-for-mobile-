"""Command-line entry point for Q-Mini quantization diagnostics."""

from qmini.quantization_ablation import main, run_ablation

__all__ = ["main", "run_ablation"]

if __name__ == "__main__":
    raise SystemExit(main())
