from __future__ import annotations
import argparse, json, platform, shutil, sys
import torch
from .config import QMiniConfig
from .data import prepare_token_streams
from .generate import generate, interactive_chat, load_model
from .evaluate import (
    emit_report, evaluate_language_model, evaluate_task_file,
    evaluate_conversation_file,
)
from .model import QMiniMoE


def _validate_thinking_steps(value: int | None, max_steps: int) -> None:
    if value is not None and not 1 <= value <= max_steps:
        raise ValueError(
            f"--thinking-steps must be 1..{max_steps} for this model configuration."
        )


def main() -> int:
    parser = argparse.ArgumentParser(prog="qmini", description="Q-Mini model, dataset and training pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="inspect model configuration and runtime")
    audit.add_argument("--config", default="configs/qmini_light_0.7m.json")
    def do_audit(args):
        cfg = QMiniConfig.from_json(args.config)
        parameters = cfg.estimated_parameter_count()
        print("Q-MINI CONFIG AUDIT"); print("=" * 60)
        print("Python:", platform.python_version(), "PyTorch:", torch.__version__)
        print("Platform:", platform.platform(), "CUDA:", torch.cuda.is_available())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {p.name} ({p.total_memory/1024**3:.1f} GiB)")
        print("Estimated parameters:", f"{parameters:,}")
        print("FP32 params + gradients + AdamW state lower bound:",
              f"{cfg.estimated_adamw_training_state_bytes():,} bytes")
        print("Architecture: causal depthwise convolution + gated diagonal SSM:",
              "enabled" if cfg.use_state_space else "disabled")
        print("Sparse routing:", f"Top-{cfg.top_k} across {cfg.n_experts} experts")
        print("Main receptive field:", cfg.main_receptive_field(),
              "tokens; max context:", cfg.max_context)
        print("Shared refinement passes:", cfg.thinking_steps,
              "/", cfg.max_thinking_steps)
        print("C++ compiler:", shutil.which("clang++") or shutil.which("g++") or "not found")
        from .native import native_kernel_status
        native = native_kernel_status()
        print("CPU projection backend:", native["backend"])
        print("ARM NEON:", native["uses_neon"])
        print("Native library:", native["path"])
        print("Attention/QKV: absent by design")
        print("Tensor allocation: skipped (safe config-only audit)")
        return 0
    audit.set_defaults(func=do_audit)

    preflight = commands.add_parser(
        "preflight",
        help="dry-run the random-initialized model graph without loading data or updating weights",
    )
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    preflight.add_argument("--sequence-length", type=int, default=8)
    preflight.add_argument("--max-parameters", type=int, default=250_000_000,
                           help="refuse to materialize larger configs unless explicitly raised")
    preflight.add_argument("--no-backward", action="store_true",
                           help="check forward shapes only; by default finite gradients are also checked")
    def do_preflight(args):
        from .preflight import run_architecture_preflight
        report = run_architecture_preflight(
            QMiniConfig.from_json(args.config),
            device=args.device,
            sequence_length=args.sequence_length,
            max_parameters=args.max_parameters,
            check_backward=not args.no_backward,
        )
        print(json.dumps(report, indent=2))
        return 0
    preflight.set_defaults(func=do_preflight)

    initial_build = commands.add_parser(
        "build-initial",
        help="build an untrained Q2-v2 initial brain without dense full-model allocation",
    )
    initial_build.add_argument("--config", required=True)
    initial_build.add_argument("--output-dir", required=True)
    initial_build.add_argument(
        "--source-revision",
        default=None,
        help="optional Git commit SHA recorded in the initial artifact manifest",
    )
    def do_build_initial(args):
        from .initial_build import build_initial_brain
        report = build_initial_brain(
            QMiniConfig.from_json(args.config),
            args.output_dir,
            source_revision=args.source_revision,
        )
        print(json.dumps(report, indent=2))
        return 0
    initial_build.set_defaults(func=do_build_initial)

    tok = commands.add_parser("tokenizer", help="train a local BPE tokenizer")
    tok.add_argument("--data-dir", nargs="+", required=True); tok.add_argument("--output", required=True)
    tok.add_argument("--vocab-size", type=int, default=4096); tok.add_argument("--min-frequency", type=int, default=2)
    tok.add_argument("--base-config", default="configs/qmini_light_0.7m.json"); tok.add_argument("--config-output", required=True)
    tok.add_argument("--valid-fraction", type=float, default=0.02,
        help="use the same document holdout fraction as data preparation")
    tok.add_argument(
        "--near-duplicate-hamming-distance", type=int, choices=range(0, 4), default=0,
        help="group near-duplicate documents before tokenizer split; 0 disables SimHash grouping",
    )
    tok.add_argument(
        "--source-metadata",
        help="JSON source/license manifest in qmini-source-provenance-v1 format",
    )
    tok.add_argument(
        "--release-mode", action="store_true",
        help="require explicit source provenance/licenses and block records with likely PII/secrets",
    )
    tok.add_argument("--overwrite", action="store_true")
    def do_tokenizer(args):
        from .tokenizer import train_tokenizer
        t = train_tokenizer(
            args.data_dir, args.output, args.vocab_size, args.min_frequency,
            args.overwrite, validation_fraction=args.valid_fraction,
            near_duplicate_distance=args.near_duplicate_hamming_distance,
            source_metadata_path=args.source_metadata,
            release_mode=args.release_mode,
        )
        cfg = QMiniConfig.from_json(args.base_config); cfg.vocab_size = t.get_vocab_size(); cfg.validate(); cfg.save(args.config_output)
        print("Tokenizer:", args.output, "vocab:", cfg.vocab_size, "config:", args.config_output); return 0
    tok.set_defaults(func=do_tokenizer)
    prep = commands.add_parser("prepare", help="prepare immutable train and validation token streams")
    prep.add_argument("--data-dir", nargs="+", required=True); prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--output-dir", required=True); prep.add_argument("--valid-fraction", type=float, default=0.02)
    prep.add_argument("--loss-mask", choices=("auto", "all", "assistant"), default="auto",
        help="auto: assistant-only targets for chats/Q&A, full LM loss for plain text")
    prep.add_argument(
        "--include-absolute-source-paths", action="store_true",
        help="explicitly include machine-specific absolute source paths in dataset.json",
    )
    prep.add_argument(
        "--near-duplicate-hamming-distance", type=int, choices=range(0, 4), default=0,
        help="keep similar documents in the same split; 0 uses exact-deduplication only",
    )
    prep.add_argument(
        "--source-metadata",
        help="JSON source/license manifest in qmini-source-provenance-v1 format",
    )
    prep.add_argument(
        "--release-mode", action="store_true",
        help="require explicit source provenance/licenses and block records with likely PII/secrets",
    )
    prep.set_defaults(func=lambda a: (
        print(json.dumps(prepare_token_streams(
            a.data_dir, a.tokenizer, a.output_dir, a.valid_fraction, a.loss_mask,
            include_absolute_source_paths=a.include_absolute_source_paths,
            near_duplicate_distance=a.near_duplicate_hamming_distance,
            source_metadata_path=a.source_metadata,
            release_mode=a.release_mode,
        ), indent=2)) or 0
    ))
    train = commands.add_parser("train", help="train from scratch or resume a checkpoint")
    for flag in ("config", "train-bin", "valid-bin", "tokenizer", "output-dir"): train.add_argument("--" + flag, required=True)
    train.add_argument("--steps", type=int, required=True, help="total optimizer-step target; increase it to continue a run after resume"); train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--seq-len", type=int, default=512); train.add_argument("--grad-accum", type=int, default=8)
    train.add_argument("--lr", type=float, default=3e-4); train.add_argument("--weight-decay", type=float, default=0.1)
    train.add_argument("--eval-every", type=int, default=250); train.add_argument("--save-every", type=int, default=250)
    train.add_argument("--eval-batches", type=int, default=20); train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--init-from",
        help=(
            "initialize a new fine-tuning run from final_model/model.pt or "
            "a packed final_model/model.qmini.npz; optimizer state is reset"
        ),
    )
    train.add_argument(
        "--init-from-initial",
        help=(
            "start QAT pretraining from a verified step-zero Q2-v2 initial brain; "
            "this is distinct from SFT --init-from"
        ),
    )
    train.add_argument(
        "--train-mask-bin",
        help="aligned uint8 target mask; required for prepared auto/assistant datasets",
    )
    train.add_argument(
        "--valid-mask-bin",
        help="aligned uint8 validation target mask; required for prepared auto/assistant datasets",
    )
    train.add_argument("--extend-schedule", action="store_true",
        help="explicitly extend the saved learning-rate schedule")
    train.add_argument(
        "--checkpoint-mirror-dir",
        default=None,
        help="verified Drive checkpoint store; updates a single progress counter after each saved checkpoint",
    )
    train.add_argument(
        "--checkpoint-session-id",
        default=None,
        help="active training lease/session identifier required with --checkpoint-mirror-dir",
    )
    train.add_argument("--allow-legacy-resume", action="store_true",
        help="resume a checkpoint without data fingerprints after manual verification")
    train.add_argument("--allow-dataset-change", action="store_true",
                       help="resume with a different dataset after an explicit, audited dataset switch; re-baselines only the data fingerprints in the resume contract")
    def do_train(a):
        from .train import train_model
        train_model(QMiniConfig.from_json(a.config), a.train_bin, a.valid_bin, a.output_dir,
            a.steps, a.batch_size, a.seq_len, a.grad_accum, a.lr, a.weight_decay,
            a.eval_every, a.save_every, a.eval_batches, a.resume, a.tokenizer, None,
            a.extend_schedule, a.allow_legacy_resume, a.train_mask_bin, a.valid_mask_bin,
            init_from=a.init_from, init_from_initial=a.init_from_initial,
            checkpoint_mirror_dir=a.checkpoint_mirror_dir,
            checkpoint_session_id=a.checkpoint_session_id, allow_dataset_change=a.allow_dataset_change); return 0
    train.set_defaults(func=do_train)
    gen = commands.add_parser("generate", help="generate from a trained model")
    gen.add_argument("--model-dir", required=True); gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-new-tokens", type=int, default=128); gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top-k", type=int, default=40); gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--thinking-steps", type=int); gen.add_argument("--device", default=None)
    gen.add_argument("--frequency-penalty", type=float, default=0.0)
    gen.add_argument("--presence-penalty", type=float, default=0.0)
    gen.add_argument(
        "--serialized-prompt",
        action="store_true",
        help="Trust Q-Mini control-token markers in the prompt (unsafe for untrusted text).",
    )
    def do_generate(a):
        model, tokenizer, _ = load_model(a.model_dir, a.device)
        try:
            _validate_thinking_steps(a.thinking_steps, model.config.max_thinking_steps)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if a.serialized_prompt:
            print(
                "WARNING: --serialized-prompt trusts role/control-token markers in "
                "the supplied prompt. Use only with prompts produced by Q-Mini's "
                "trusted serializer.",
                file=sys.stderr,
            )
        print(generate(
            model, tokenizer, a.prompt, a.max_new_tokens, a.temperature,
            a.top_k, a.top_p, a.thinking_steps,
            serialized_prompt=a.serialized_prompt,
            frequency_penalty=a.frequency_penalty,
            presence_penalty=a.presence_penalty,
        )); return 0
    gen.set_defaults(func=do_generate)
    verify_export = commands.add_parser("verify-export", help="validate a packed Q-Mini NPZ export")
    verify_export.add_argument("--path", required=True)
    def do_verify_export(a):
        from .export import verify_packed_export
        print(json.dumps(verify_packed_export(a.path), indent=2)); return 0
    verify_export.set_defaults(func=do_verify_export)
    memory = commands.add_parser("memory", help="manage explicit local SQLite memories")
    memory.add_argument("--db", required=True, help="local SQLite database path")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)

    memory_add = memory_commands.add_parser("add", help="save a note explicitly")
    memory_add.add_argument("--text", required=True)
    memory_add.add_argument("--tag", action="append", default=[])
    def do_memory_add(a):
        from .memory import LocalMemoryStore
        store = LocalMemoryStore(a.db)
        print(json.dumps(store.remember(a.text, tags=a.tag), ensure_ascii=False, indent=2))
        return 0
    memory_add.set_defaults(func=do_memory_add)

    memory_search = memory_commands.add_parser("search", help="search saved notes")
    memory_search.add_argument("--query", default="")
    memory_search.add_argument("--limit", type=int, default=8)
    def do_memory_search(a):
        from .memory import LocalMemoryStore
        print(json.dumps(
            LocalMemoryStore(a.db).search(a.query, a.limit),
            ensure_ascii=False, indent=2
        ))
        return 0
    memory_search.set_defaults(func=do_memory_search)

    memory_forget = memory_commands.add_parser("forget", help="delete one saved note by ID")
    memory_forget.add_argument("--id", type=int, required=True)
    def do_memory_forget(a):
        from .memory import LocalMemoryStore
        removed = LocalMemoryStore(a.db).forget(a.id)
        print("deleted" if removed else "not-found")
        return 0
    memory_forget.set_defaults(func=do_memory_forget)

    memory_export = memory_commands.add_parser("export", help="export memories to JSON")
    memory_export.add_argument("--output", required=True)
    def do_memory_export(a):
        from .memory import LocalMemoryStore
        target = LocalMemoryStore(a.db).export_json(a.output)
        print(target)
        return 0
    memory_export.set_defaults(func=do_memory_export)

    chat = commands.add_parser("chat", help="interactive terminal chat")
    chat.add_argument("--model-dir", required=True)
    chat.add_argument("--max-new-tokens", type=int, default=160)
    chat.add_argument("--memory-db", default=None,
        help="opt-in local SQLite memory database; writes require /remember")
    chat.set_defaults(func=lambda a: (
        interactive_chat(a.model_dir, a.max_new_tokens, a.memory_db) or 0
    ))

    eval_lm = commands.add_parser("eval-lm", help="evaluate held-out language-model loss and expert usage")
    eval_lm.add_argument("--model-dir", required=True)
    eval_lm.add_argument("--tokens", required=True)
    eval_lm.add_argument("--mask", default=None)
    eval_lm.add_argument("--batch-size", type=int, default=2)
    eval_lm.add_argument("--seq-len", type=int, default=512)
    eval_lm.add_argument("--batches", type=int, default=100)
    eval_lm.add_argument("--seed", type=int, default=42)
    eval_lm.add_argument("--device", default=None)
    eval_lm.add_argument("--output", default=None)
    def do_eval_lm(a):
        model, _tokenizer, _device = load_model(a.model_dir, a.device)
        report = evaluate_language_model(
            model, a.tokens, a.mask, a.batch_size, a.seq_len, a.batches, a.seed
        )
        emit_report(report, a.output)
        return 0
    eval_lm.set_defaults(func=do_eval_lm)

    eval_tasks = commands.add_parser("eval-tasks", help="evaluate a JSONL prompt/reference set")
    eval_tasks.add_argument("--model-dir", required=True)
    eval_tasks.add_argument("--dataset", required=True)
    eval_tasks.add_argument("--max-new-tokens", type=int, default=128)
    eval_tasks.add_argument("--limit", type=int, default=None)
    eval_tasks.add_argument("--thinking-steps", type=int, default=None)
    eval_tasks.add_argument("--device", default=None)
    eval_tasks.add_argument("--output", default=None)
    def do_eval_tasks(a):
        model, tokenizer, _device = load_model(a.model_dir, a.device)
        try:
            _validate_thinking_steps(a.thinking_steps, model.config.max_thinking_steps)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        report = evaluate_task_file(
            model, tokenizer, a.dataset, a.max_new_tokens, a.limit, a.thinking_steps
        )
        emit_report(report, a.output)
        return 0
    eval_tasks.set_defaults(func=do_eval_tasks)

    eval_conversations = commands.add_parser(
        "eval-conversations",
        help="evaluate multi-turn conversation continuity using a JSONL task file",
    )
    eval_conversations.add_argument("--model-dir", required=True)
    eval_conversations.add_argument("--dataset", required=True)
    eval_conversations.add_argument("--max-new-tokens", type=int, default=128)
    eval_conversations.add_argument("--limit", type=int, default=None)
    eval_conversations.add_argument("--thinking-steps", type=int, default=None)
    eval_conversations.add_argument("--device", default=None)
    eval_conversations.add_argument("--output", default=None)
    def do_eval_conversations(a):
        model, tokenizer, _device = load_model(a.model_dir, a.device)
        try:
            _validate_thinking_steps(a.thinking_steps, model.config.max_thinking_steps)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        report = evaluate_conversation_file(
            model, tokenizer, a.dataset, a.max_new_tokens, a.limit, a.thinking_steps
        )
        emit_report(report, a.output)
        return 0
    eval_conversations.set_defaults(func=do_eval_conversations)

    args = parser.parse_args()
    result = args.func(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
