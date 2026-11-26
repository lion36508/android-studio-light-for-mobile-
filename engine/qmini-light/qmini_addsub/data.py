from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


OASST_REQUIRED_FIELDS = {
    "message_id", "parent_id", "text", "role", "lang", "review_result",
    "deleted", "rank", "synthetic", "message_tree_id", "tree_state",
}

EMAIL_RE = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRET_RE = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})\b")
PHONE_CONTEXT_RE = re.compile(
    r"(?i)\b(?:phone|mobile|cell(?:ular)?|telephone|whatsapp|call|text)\b"
    r"[^\d\n]{0,32}\+?\d[\d\s().-]{7,}\d"
)
SPECIAL_TOKEN_LITERALS = (
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
    "<bos>", "<eos>", "<pad>", "<unk>",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_training_text(
    value: Any,
    min_chars: int = 2,
    max_chars: int = 32768,
) -> tuple[str, str]:
    """Normalize training text and return a stable rejection reason code.

    This is not a language detector. OASST1's explicit lang field and the
    pinned English Dolly source provide language provenance.
    """
    if not isinstance(value, str):
        return "", "not_text"
    text = unicodedata.normalize("NFC", value)
    text = text.replace("@-@", "-")
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        char for char in text
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < min_chars:
        return text, "too_short"
    if len(text) > max_chars:
        return text, "too_long"
    if not any(char.isalpha() for char in text):
        return text, "no_letters"
    if EMAIL_RE.search(text):
        return text, "email"
    if SECRET_RE.search(text):
        return text, "credential"
    if PHONE_CONTEXT_RE.search(text):
        return text, "phone_contact"
    return text, "accepted"


def is_clean_english_text(value: Any, min_chars: int = 2) -> bool:
    """Compatibility wrapper for text hygiene, not language identification."""
    return normalize_training_text(value, min_chars=min_chars)[1] == "accepted"


def escape_special_token_literals(text: str) -> str:
    """Prevent literal dataset text from becoming tokenizer control IDs."""
    value = text
    for token in SPECIAL_TOKEN_LITERALS:
        value = value.replace(token, token.replace("<", "<\u200b", 1))
    return value


def accept_oasst1_message(row: dict[str, Any]) -> tuple[bool, str]:
    """Fail-closed source filter: reviewed human English OASST1 only."""
    if not isinstance(row, dict):
        return False, "not_record"
    missing = OASST_REQUIRED_FIELDS - set(row)
    if missing:
        return False, "missing_fields"
    if row["lang"] != "en":
        return False, "non_english"
    if row["tree_state"] != "ready_for_export":
        return False, "not_ready"
    if row["review_result"] is not True:
        return False, "unreviewed"
    if row["deleted"] is not False:
        return False, "deleted"
    if row["synthetic"] is not False:
        return False, "synthetic"
    if not isinstance(row["message_id"], str) or not row["message_id"].strip():
        return False, "invalid_message_id"
    if not isinstance(row["message_tree_id"], str) or not row["message_tree_id"].strip():
        return False, "invalid_tree_id"
    role = row["role"]
    if role == "assistant":
        rank = row["rank"]
        if rank is None or isinstance(rank, bool):
            return False, "missing_rank"
        try:
            if int(rank) != 0:
                return False, "non_top_rank"
        except (TypeError, ValueError):
            return False, "invalid_rank"
    elif role != "prompter":
        return False, "other_role"
    if not is_clean_english_text(row["text"]):
        return False, "empty_or_sensitive"
    return True, "accepted"


def stable_split(group_id: str, seed: str = "qmini-addsub-v1") -> str:
    """Deterministically split whole documents/trees without row leakage."""
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("A non-empty group ID is required for leakage-safe splitting.")
    digest = hashlib.sha256((seed + ":" + group_id).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> tuple[int, str]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count, sha256_file(destination)
