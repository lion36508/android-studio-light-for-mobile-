from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

from qmini_addsub.config import AddSubConfig
from qmini_addsub.data import escape_special_token_literals, normalize_training_text
from qmini_addsub.train import sha256_file


def append_sft_training_text(target, sft_path: str | Path, approved_sources: set[str]) -> dict[str, int]:
    """Append only manifest-approved human-authored SFT train messages to BPE text."""
    path = Path(sft_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(approved_sources, set) or not approved_sources or any(
        not isinstance(value, str) or not value.strip() for value in approved_sources
    ):
        raise ValueError("A non-empty set of approved SFT sources is required.")

    conversations = 0
    messages_written = 0
    sampled_bytes = 0
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid SFT JSONL.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: SFT record must be an object.")
            if record.get("source") not in approved_sources:
                raise ValueError(f"{path}:{line_number}: SFT source is not approved by the manifest.")
            if record.get("human_authored") is not True or record.get("synthetic") is not False:
                raise ValueError(f"{path}:{line_number}: synthetic or unverified SFT text is forbidden.")
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_number}: SFT messages must contain a conversation.")
            roles = [item.get("role") if isinstance(item, dict) else None for item in messages]
            if roles[0] != "user" or roles[-1] != "assistant":
                raise ValueError(f"{path}:{line_number}: SFT conversation must start with user and end with assistant.")
            if any(role not in {"user", "assistant"} for role in roles) or any(
                roles[index] == roles[index - 1] for index in range(1, len(roles))
            ):
                raise ValueError(f"{path}:{line_number}: SFT roles must alternate between user and assistant.")

            for message in messages:
                cleaned, reason = normalize_training_text(message.get("content"))
                if reason != "accepted":
                    raise ValueError(f"{path}:{line_number}: SFT message rejected by cleaner ({reason}).")
                escaped = escape_special_token_literals(cleaned)
                target.write(escaped + "\n")
                sampled_bytes += len(escaped.encode("utf-8"))
                messages_written += 1
            conversations += 1

    if conversations == 0 or messages_written == 0:
        raise ValueError(f"No usable manifest-approved SFT training text found in {path}.")
    # This is prepended to every SFT example during supervised fine-tuning.
    system_prompt = "You are Q-Mini."
    target.write(system_prompt + "\n")
    sampled_bytes += len(system_prompt.encode("utf-8"))
    return {
        "conversations": conversations,
        "messages": messages_written,
        "bytes": sampled_bytes,
    }


SPECIAL_TOKENS = [
    "<unk>",
    "<pad>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the isolated Add/Sub v1 tokenizer from training-split text only."
    )
    parser.add_argument("--data-dir", default="data/qmini_addsub_v1")
    parser.add_argument("--output", default="artifacts/qmini_addsub_v1/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--pretrain-byte-budget", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pretrain = data_dir / "pretrain_train.txt"
    sft = data_dir / "sft_train.jsonl"
    manifest = data_dir / "dataset_manifest.json"
    if not pretrain.is_file() or not manifest.is_file():
        raise FileNotFoundError("Prepare or import the selected English pretraining split first.")
    data_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        data_manifest.get("format") != "qmini-addsub-data-v1"
        or data_manifest.get("language") != "English"
        or data_manifest.get("synthetic_training_records_included") != 0
    ):
        raise ValueError("Tokenizer requires a verified English, non-synthetic Add/Sub dataset manifest.")
    expected_pretrain = data_manifest.get("pretraining", {}).get("splits", {}).get("train", {})
    if expected_pretrain.get("sha256") != sha256_file(pretrain):
        raise ValueError("Pretraining train text does not match the dataset manifest.")
    instruction_splits = data_manifest.get("instruction_splits", {})
    sft_train_entry = instruction_splits.get("train")
    if sft_train_entry is not None:
        if not sft.is_file():
            raise FileNotFoundError("Manifest declares an SFT training split, but sft_train.jsonl is missing.")
        if sft_train_entry.get("path") != sft.name or sft_train_entry.get("sha256") != sha256_file(sft):
            raise ValueError("SFT train file does not match the dataset manifest.")
        for split_name in ("validation", "test"):
            entry = instruction_splits.get(split_name)
            if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
                raise ValueError(f"Manifest is missing the SFT {split_name} split.")
            split_path = data_dir / entry["path"]
            if not split_path.is_file() or sha256_file(split_path) != entry["sha256"]:
                raise ValueError(f"SFT {split_name} file does not match the dataset manifest.")
    elif sft.is_file():
        raise ValueError("SFT train file exists but is not declared in the dataset manifest.")
    if not 264 <= args.vocab_size <= 65536:
        raise ValueError("Byte-level BPE vocab_size must be between 264 and 65536 (uint16 token streams).")
    if args.pretrain_byte_budget < 1 or args.seed < 0:
        raise ValueError("Pretraining byte budget must be positive and seed non-negative.")

    # Only the training split is used. Validation/test text is never read here.
    trainer_corpus = Path(args.output).with_suffix(".training-text.tmp")
    output = Path(args.output)
    config_path = output.with_name("model_config.json")
    tokenizer_temporary = output.with_suffix(output.suffix + ".tmp")
    config_temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    if output == config_path:
        raise ValueError("Tokenizer output path must not be named model_config.json.")
    if output.exists() or config_path.exists():
        raise FileExistsError(
            "Tokenizer or model_config.json already exists. Choose a fresh output "
            "path; replacing a tokenizer used by existing checkpoints is forbidden."
        )
    trainer_corpus.parent.mkdir(parents=True, exist_ok=True)
    pretrain_source_bytes = pretrain.stat().st_size
    sample_probability = min(1.0, args.pretrain_byte_budget / max(1, pretrain_source_bytes))
    rng = random.Random(args.seed)
    pretrain_sampled_bytes = 0
    sft_sampled_bytes = 0
    try:
        with trainer_corpus.open("w", encoding="utf-8") as target:
            # Sample up to 64 MiB of pretraining training text; include the
            # complete SFT training split so chat vocabulary affects BPE merges.
            with pretrain.open("r", encoding="utf-8") as source:
                first_accepted = None
                for line in source:
                    raw = line.strip()
                    if not raw:
                        continue
                    cleaned, reason = normalize_training_text(raw, min_chars=1)
                    if reason != "accepted":
                        continue
                    escaped = escape_special_token_literals(cleaned)
                    if first_accepted is None:
                        first_accepted = escaped
                    if rng.random() > sample_probability:
                        continue
                    count = len(escaped.encode("utf-8"))
                    if pretrain_sampled_bytes + count > args.pretrain_byte_budget:
                        continue
                    target.write(escaped + "\n")
                    pretrain_sampled_bytes += count
                if pretrain_sampled_bytes == 0 and first_accepted is not None:
                    first_count = len(first_accepted.encode("utf-8"))
                    if first_count > args.pretrain_byte_budget:
                        raise RuntimeError(
                            "Pretraining byte budget is smaller than the first usable "
                            "training line; increase --pretrain-byte-budget."
                        )
                    target.write(first_accepted + "\n")
                    pretrain_sampled_bytes = first_count

            sft_stats = {"conversations": 0, "messages": 0, "bytes": 0}
            if sft_train_entry is not None:
                approved_sources = data_manifest.get("approved_instruction_sources")
                if not isinstance(approved_sources, list) or not approved_sources:
                    raise ValueError("Dataset manifest must declare approved_instruction_sources for SFT tokenizer text.")
                sft_stats = append_sft_training_text(
                    target, sft, set(approved_sources)
                )
                sft_sampled_bytes = sft_stats["bytes"]

        if pretrain_sampled_bytes == 0:
            raise RuntimeError("Tokenizer sampling produced no usable pretraining text.")

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tokenizer.decoder = ByteLevel()
        byte_alphabet = ByteLevelPreTokenizer.alphabet()
        trainer = BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=byte_alphabet,
            show_progress=True,
        )
        tokenizer.train([str(trainer_corpus)], trainer)

        for token in SPECIAL_TOKENS:
            if tokenizer.token_to_id(token) is None:
                raise RuntimeError(f"Tokenizer is missing required token: {token}")
        actual_vocab_size = tokenizer.get_vocab_size()
        if not 264 <= actual_vocab_size <= args.vocab_size:
            raise RuntimeError(
                f"Tokenizer vocabulary {actual_vocab_size} is outside the valid range "
                f"[264, {args.vocab_size}]."
            )

        model_config = AddSubConfig(vocab_size=actual_vocab_size)
        tokenizer.save(str(tokenizer_temporary))
        config_temporary.write_text(
            json.dumps(model_config.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Publish the config first and tokenizer last; roll back the config
        # if the tokenizer rename fails.
        config_temporary.replace(config_path)
        try:
            tokenizer_temporary.replace(output)
        except Exception:
            config_path.unlink(missing_ok=True)
            raise

        trained_on = ["pretrain_train.txt"]
        if sft_train_entry is not None:
            trained_on.append("sft_train.jsonl")
        print(json.dumps({
            "tokenizer": str(output),
            "model_config": str(config_path),
            "vocab_size": actual_vocab_size,
            "special_tokens": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
            "trained_on": trained_on,
            "pretrain_source_bytes": pretrain_source_bytes,
            "pretrain_sampled_bytes": pretrain_sampled_bytes,
            "sft_sampled_bytes": sft_sampled_bytes,
            "sft_conversations": sft_stats["conversations"],
            "sft_messages": sft_stats["messages"],
            "sft_tokenizer_corpus_share": sft_sampled_bytes / max(1, pretrain_sampled_bytes + sft_sampled_bytes),
            "validation_test_used": False,
        }, indent=2))
    finally:
        trainer_corpus.unlink(missing_ok=True)
        tokenizer_temporary.unlink(missing_ok=True)
        config_temporary.unlink(missing_ok=True)



if __name__ == "__main__":
    main()
