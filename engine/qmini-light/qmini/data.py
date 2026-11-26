from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlsplit

import numpy as np

from .text import escape_special_token_literals

SUPPORTED_SUFFIXES = {
    ".txt", ".text", ".md", ".markdown", ".json",
    ".jsonl", ".ndjson", ".csv", ".parquet",
}
ASSISTANT_ROLES = {"assistant", "bot", "model"}


def expand_data_paths(paths: Iterable[str | Path]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            found.append(path.resolve())
        elif path.is_dir():
            found.extend(
                item.resolve()
                for item in sorted(path.rglob("*"))
                if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
            )
        elif not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
    return sorted(set(found))


def extract_text(record: object) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, list):
        return "\n".join(
            text for item in record
            if (text := extract_text(item)).strip()
        )
    if not isinstance(record, dict):
        return ""

    messages = record.get("messages")
    if isinstance(messages, list):
        lines = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get("role", "unknown")).strip().upper()
                body = extract_text(message.get("content", message.get("text", "")))
                if body.strip():
                    lines.append(f"[{role}]\n{body}")
        if lines:
            return "\n".join(lines)

    keys = (
        "text", "content", "instruction", "input", "prompt", "question",
        "completion", "output", "answer", "response", "reasoning",
    )
    parts = [
        extract_text(record[key])
        for key in keys
        if record.get(key) is not None
    ]
    parts = [part for part in parts if part.strip()]
    if parts:
        return "\n".join(parts)

    fallback = [
        extract_text(value)
        for value in record.values()
        if isinstance(value, (str, list, dict))
    ]
    return "\n".join(part for part in fallback if part.strip())


def _iter_plain(path: Path, chunk_chars: int = 8192) -> Iterator[str]:
    """Yield bounded plain-text chunks, rejecting malformed UTF-8.

    Silently replacing invalid bytes changes the training corpus while making
    the resulting dataset appear valid. Require callers to repair or explicitly
    convert malformed source files before tokenizer fitting or preparation.
    """
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be positive.")
    buffer: list[str] = []
    size = 0
    try:
        with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
            for line in handle:
                if not line.strip():
                    if buffer:
                        yield "\n".join(buffer)
                        buffer, size = [], 0
                    continue
                buffer.append(line.rstrip())
                size += len(line)
                if size >= chunk_chars:
                    yield "\n".join(buffer)
                    buffer, size = [], 0
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Invalid UTF-8 in plain-text dataset {path}; repair or explicitly "
            "transcode the source file before training."
        ) from exc
    if buffer:
        yield "\n".join(buffer)


def _ensure_json_eof(
    handle,
    buffer: str,
    cursor: int,
    path: Path,
    chunk_chars: int,
) -> None:
    """Reject trailing JSON content while reading only bounded chunks."""
    if buffer[cursor:].strip():
        raise ValueError(f"Unexpected data after JSON array in {path}.")
    while True:
        trailing = handle.read(chunk_chars)
        if not trailing:
            return
        if trailing.strip():
            raise ValueError(f"Unexpected data after JSON array in {path}.")


def _iter_json_array(handle, path: Path, chunk_chars: int = 65536) -> Iterator[object]:
    """Incrementally decode a top-level JSON array with bounded input buffering."""
    opening = handle.read(1)
    while opening and opening.isspace():
        opening = handle.read(1)
    if opening != "[":
        raise ValueError(f"Expected a top-level JSON array in {path}.")
    decoder = json.JSONDecoder()
    buffer = handle.read(chunk_chars)
    cursor = 0
    eof = len(buffer) == 0
    expect_value = True
    values_seen = False

    while True:
        while cursor >= len(buffer) and not eof:
            chunk = handle.read(chunk_chars)
            if chunk:
                buffer += chunk
            else:
                eof = True

        while cursor < len(buffer) and buffer[cursor].isspace():
            cursor += 1

        if cursor >= len(buffer):
            if eof:
                raise ValueError(f"JSON array in {path} ended before its closing bracket.")
            if cursor:
                buffer = buffer[cursor:]
                cursor = 0
            continue

        char = buffer[cursor]
        if expect_value:
            if char == "]":
                if values_seen:
                    raise ValueError(f"Trailing comma in JSON array in {path}.")
                cursor += 1
                _ensure_json_eof(handle, buffer, cursor, path, chunk_chars)
                return
            try:
                value, end = decoder.raw_decode(buffer, cursor)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError(
                        f"Invalid or truncated JSON array in {path} near character {cursor}: {exc.msg}"
                    ) from exc
                if cursor:
                    buffer = buffer[cursor:]
                    cursor = 0
                chunk = handle.read(chunk_chars)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            yield value
            values_seen = True
            cursor = end
            expect_value = False
        else:
            if char == ",":
                cursor += 1
                expect_value = True
            elif char == "]":
                cursor += 1
                _ensure_json_eof(handle, buffer, cursor, path, chunk_chars)
                return
            else:
                raise ValueError(
                    f"Expected ',' or ']' after a JSON value in {path} at character {cursor}."
                )

        if cursor >= chunk_chars:
            buffer = buffer[cursor:]
            cursor = 0


def _iter_json_file(path: Path) -> Iterator[object]:
    with path.open("r", encoding="utf-8-sig") as handle:
        first_non_whitespace = ""
        while True:
            char = handle.read(1)
            if not char or not char.isspace():
                first_non_whitespace = char
                break
        handle.seek(0)
        if first_non_whitespace == "[":
            yield from _iter_json_array(handle, path)
            return
        payload = json.load(handle)
        yield from (payload if isinstance(payload, list) else [payload])


def iter_source_records(paths: Iterable[str | Path]) -> Iterator[object]:
    """Yield source records without flattening away chat roles."""
    for path in expand_data_paths(paths):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".text", ".md", ".markdown"}:
            yield from _iter_plain(path)
        elif suffix == ".json":
            yield from _iter_json_file(path)
        elif suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {path} at line {line_number}: {exc}"
                        ) from exc
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                yield from csv.DictReader(handle)
        elif suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError(
                    "Parquet input requires pyarrow. Install with pip install pyarrow."
                ) from exc
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=512):
                columns = batch.to_pydict()
                for index in range(batch.num_rows):
                    yield {key: values[index] for key, values in columns.items()}


def iter_text_records(paths: Iterable[str | Path]) -> Iterator[str]:
    for record in iter_source_records(paths):
        text = extract_text(record).strip()
        if text:
            yield text


def _encode_dataset_text(text: str) -> bytes:
    """Encode dataset text without silently replacing invalid Unicode surrogates."""
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "Dataset text contains an invalid Unicode surrogate; repair the source "
            "record instead of silently changing its content."
        ) from exc


def _validate_near_duplicate_distance(distance: int) -> None:
    if isinstance(distance, bool) or not isinstance(distance, int) or not 0 <= distance <= 3:
        raise ValueError("near_duplicate_distance must be an integer from 0 to 3.")


def _simhash64(text: str) -> bytes:
    """Create a deterministic 64-bit SimHash from word trigrams."""
    tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    if not tokens:
        tokens = [text.casefold()]
    if len(tokens) < 3:
        shingles = [" ".join(tokens)]
    else:
        shingles = [
            " ".join(tokens[index:index + 3])
            for index in range(len(tokens) - 2)
        ]

    votes = [0] * 64
    for shingle in set(shingles):
        digest = hashlib.blake2b(
            _encode_dataset_text(shingle), digest_size=8
        ).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            votes[bit] += 1 if value & (1 << bit) else -1
    signature = sum(1 << bit for bit, vote in enumerate(votes) if vote >= 0)
    return signature.to_bytes(8, "big")


def _create_document_group_index(connection: sqlite3.Connection, distance: int) -> None:
    _validate_near_duplicate_distance(distance)
    connection.execute("CREATE TABLE documents (digest BLOB PRIMARY KEY, simhash BLOB)")
    connection.execute(
        "CREATE TABLE document_groups (digest BLOB PRIMARY KEY, parent BLOB NOT NULL)"
    )
    if distance:
        connection.execute(
            "CREATE TABLE simhash_bands (band_index INTEGER, band_value INTEGER, digest BLOB, "
            "PRIMARY KEY (band_index, band_value, digest))"
        )
        connection.execute(
            "CREATE INDEX simhash_band_lookup ON simhash_bands(band_index, band_value)"
        )


def _find_document_group_root(connection: sqlite3.Connection, digest: bytes) -> bytes:
    current = digest
    path: list[bytes] = []
    while True:
        row = connection.execute(
            "SELECT parent FROM document_groups WHERE digest=?", (current,)
        ).fetchone()
        if row is None:
            raise RuntimeError("A document is missing from the duplicate-group index.")
        parent = bytes(row[0])
        if parent == current:
            root = current
            break
        path.append(current)
        current = parent
    for member in path:
        connection.execute(
            "UPDATE document_groups SET parent=? WHERE digest=?", (root, member)
        )
    return root


def _union_document_groups(
    connection: sqlite3.Connection,
    first: bytes,
    second: bytes,
) -> None:
    root_a = _find_document_group_root(connection, first)
    root_b = _find_document_group_root(connection, second)
    if root_a == root_b:
        return
    parent, child = sorted((root_a, root_b))
    connection.execute(
        "UPDATE document_groups SET parent=? WHERE digest=?", (parent, child)
    )


def _register_document(
    connection: sqlite3.Connection,
    normalized_text: str,
    near_duplicate_distance: int,
) -> tuple[bytes, bool]:
    digest = hashlib.sha256(
        _encode_dataset_text(normalized_text)
    ).digest()
    signature = _simhash64(normalized_text) if near_duplicate_distance else None
    cursor = connection.execute(
        "INSERT OR IGNORE INTO documents(digest, simhash) VALUES (?, ?)",
        (digest, signature),
    )
    if cursor.rowcount == 0:
        return digest, False

    connection.execute(
        "INSERT INTO document_groups(digest, parent) VALUES (?, ?)",
        (digest, digest),
    )
    if not near_duplicate_distance:
        return digest, True

    signature_value = int.from_bytes(signature, "big")
    candidates: set[bytes] = set()
    # Four 16-bit LSH bands guarantee that two signatures at Hamming distance
    # <= 3 share at least one band (the three changed bits can affect at most
    # three of the four bands).
    for band_index in range(4):
        band_value = (signature_value >> (band_index * 16)) & 0xFFFF
        rows = connection.execute(
            "SELECT digest FROM simhash_bands WHERE band_index=? AND band_value=?",
            (band_index, band_value),
        ).fetchall()
        candidates.update(bytes(row[0]) for row in rows)

    for candidate in candidates:
        row = connection.execute(
            "SELECT simhash FROM documents WHERE digest=?", (candidate,)
        ).fetchone()
        if row is None or row[0] is None:
            continue
        existing_value = int.from_bytes(bytes(row[0]), "big")
        if (signature_value ^ existing_value).bit_count() <= near_duplicate_distance:
            _union_document_groups(connection, digest, candidate)

    for band_index in range(4):
        band_value = (signature_value >> (band_index * 16)) & 0xFFFF
        connection.execute(
            "INSERT INTO simhash_bands(band_index, band_value, digest) VALUES (?, ?, ?)",
            (band_index, band_value, digest),
        )
    return digest, True


def _finalize_document_group_splits(
    connection: sqlite3.Connection,
    valid_fraction: float,
) -> dict[str, int]:
    connection.execute(
        "CREATE TABLE group_splits (root BLOB PRIMARY KEY, is_valid INTEGER NOT NULL)"
    )
    for (digest_value,) in connection.execute("SELECT digest FROM documents"):
        root = _find_document_group_root(connection, bytes(digest_value))
        score = int.from_bytes(root[:8], "big") / (2**64)
        connection.execute(
            "INSERT OR IGNORE INTO group_splits(root, is_valid) VALUES (?, ?)",
            (root, int(score < valid_fraction)),
        )

    group_count = int(connection.execute(
        "SELECT COUNT(*) FROM group_splits"
    ).fetchone()[0])
    if group_count < 2:
        raise ValueError(
            "Near-duplicate grouping left fewer than two independent document groups. "
            "Lower --near-duplicate-distance or add more distinct documents."
        )
    valid_count = int(connection.execute(
        "SELECT COUNT(*) FROM group_splits WHERE is_valid=1"
    ).fetchone()[0])
    if valid_count == 0:
        chosen = connection.execute(
            "SELECT root FROM group_splits ORDER BY root LIMIT 1"
        ).fetchone()[0]
        connection.execute(
            "UPDATE group_splits SET is_valid=1 WHERE root=?", (chosen,)
        )
    elif valid_count == group_count:
        chosen = connection.execute(
            "SELECT root FROM group_splits ORDER BY root DESC LIMIT 1"
        ).fetchone()[0]
        connection.execute(
            "UPDATE group_splits SET is_valid=0 WHERE root=?", (chosen,)
        )
    connection.commit()
    return {
        "groups": group_count,
        "valid_groups": int(connection.execute(
            "SELECT COUNT(*) FROM group_splits WHERE is_valid=1"
        ).fetchone()[0]),
    }


def _document_is_validation(
    connection: sqlite3.Connection,
    digest: bytes,
) -> bool:
    root = _find_document_group_root(connection, digest)
    row = connection.execute(
        "SELECT is_valid FROM group_splits WHERE root=?", (root,)
    ).fetchone()
    if row is None:
        raise RuntimeError("A source document has no deterministic train/validation group.")
    return bool(row[0])


def iter_training_text_records(
    paths: Iterable[str | Path],
    valid_fraction: float = 0.02,
    near_duplicate_distance: int = 0,
) -> Iterator[str]:
    """Yield unique training documents after grouping near-duplicates on disk.

    Optional SimHash grouping keeps highly similar documents in one split.
    A distance of zero preserves exact-document deduplication only.
    """
    if not 0.0 < valid_fraction < 0.5:
        raise ValueError("valid_fraction must be greater than 0 and below 0.5.")
    _validate_near_duplicate_distance(near_duplicate_distance)
    source_paths = expand_data_paths(paths)
    if not source_paths:
        raise FileNotFoundError("No supported dataset files were found.")

    with tempfile.TemporaryDirectory(prefix="qmini-tokenizer-") as temp_dir:
        database_path = Path(temp_dir) / "split.sqlite3"
        # Hugging Face tokenizers may resume this single-consumer Python iterator
        # from a worker thread. SQLite work is sequential (the generator cannot
        # be entered concurrently), so allow the connection to migrate between
        # threads while retaining SQLite's normal locking/transaction semantics.
        connection = sqlite3.connect(database_path, check_same_thread=False)
        yielded = 0
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            _create_document_group_index(connection, near_duplicate_distance)
            connection.execute("CREATE TABLE written (digest BLOB PRIMARY KEY)")

            with connection:
                for document in iter_text_records(source_paths):
                    normalized = document.strip()
                    if len(normalized) < 2:
                        continue
                    _register_document(
                        connection, normalized, near_duplicate_distance
                    )

            unique_count = int(connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0])
            if unique_count < 2:
                raise ValueError(
                    "No unique training documents: at least two unique documents are required "
                    "for a separated training/validation tokenizer split."
                )
            grouping = _finalize_document_group_splits(connection, valid_fraction)
            if grouping["groups"] < 2:
                raise ValueError("Tokenizer split requires at least two independent groups.")

            for document in iter_text_records(source_paths):
                normalized = document.strip()
                if len(normalized) < 2:
                    continue
                digest = hashlib.sha256(
                    _encode_dataset_text(normalized)
                ).digest()
                if connection.execute(
                    "SELECT 1 FROM documents WHERE digest=?", (digest,)
                ).fetchone() is None:
                    raise RuntimeError("Source data changed during tokenizer training.")
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO written(digest) VALUES (?)", (digest,)
                )
                if inserted.rowcount == 0 or _document_is_validation(connection, digest):
                    continue
                yielded += 1
                if yielded % 10000 == 0:
                    connection.commit()
                yield normalized

            if yielded == 0:
                raise ValueError("No unique training documents remain for tokenizer training.")
        finally:
            connection.close()
def _record_segments(record: object) -> tuple[list[tuple[str, bool]], bool, bool, bool]:
    """Return segments, structured flag, any-assistant flag and final-assistant flag."""
    if isinstance(record, dict):
        messages = record.get("messages")
        if isinstance(messages, list):
            segments: list[tuple[str, bool]] = []
            has_assistant = False
            final_message_is_assistant = False
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "unknown")).strip().lower()
                body = escape_special_token_literals(
                    extract_text(message.get("content", message.get("text", ""))).strip()
                )
                if not body:
                    continue
                is_assistant = role in ASSISTANT_ROLES
                has_assistant = has_assistant or is_assistant
                final_message_is_assistant = is_assistant
                if role in ASSISTANT_ROLES:
                    control_role = "assistant"
                elif role in {"user"}:
                    control_role = "user"
                elif role in {"system", "developer"}:
                    control_role = "system"
                elif role in {"tool", "function"}:
                    control_role = "tool"
                else:
                    control_role = "other"
                segments.append((f"<|{control_role}|>", is_assistant))
                segments.append((body + "\n", is_assistant))
                segments.append(("<|end|>\n", is_assistant))
            if segments:
                return segments, True, has_assistant, final_message_is_assistant

        user_fields = ("instruction", "input", "prompt", "question")
        assistant_fields = ("completion", "output", "answer", "response")
        user_text = "\n".join(
            extract_text(record[key]).strip()
            for key in user_fields
            if record.get(key) is not None and extract_text(record[key]).strip()
        )
        assistant_text = "\n".join(
            extract_text(record[key]).strip()
            for key in assistant_fields
            if record.get(key) is not None and extract_text(record[key]).strip()
        )
        user_text = escape_special_token_literals(user_text)
        assistant_text = escape_special_token_literals(assistant_text)
        if user_text and assistant_text:
            return [
                ("<|user|>", False),
                (user_text + "\n", False),
                ("<|end|>\n", False),
                ("<|assistant|>", True),
                (assistant_text + "\n", True),
                ("<|end|>\n", True),
            ], True, True, True

    text = escape_special_token_literals(extract_text(record).strip())
    if not text:
        return [], False, False, False
    return [(text, True)], False, True, True


def _encode_record(
    record: object,
    tokenizer,
    bos_id: int,
    eos_id: int,
    loss_mask_mode: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    segments, structured, has_assistant, final_message_is_assistant = _record_segments(record)
    if not segments:
        return None

    if loss_mask_mode == "assistant" and (not structured or not has_assistant):
        return None

    ids = [bos_id]
    mask = [0]
    for segment, assistant_target in segments:
        encoded = tokenizer.encode(segment).ids
        if not encoded:
            continue
        ids.extend(encoded)
        if loss_mask_mode == "all" or not structured:
            mask.extend([1] * len(encoded))
        elif loss_mask_mode == "assistant":
            mask.extend([int(assistant_target)] * len(encoded))
        else:  # auto: train assistant turns only for structured chat/Q&A records.
            mask.extend([int(assistant_target)] * len(encoded))

    eos_is_target = (
        loss_mask_mode == "all"
        or not structured
        or (loss_mask_mode in {"auto", "assistant"} and final_message_is_assistant)
    )
    ids.append(eos_id)
    mask.append(int(eos_is_target))

    if not any(mask):
        return None
    return (
        np.asarray(ids, dtype=np.uint16),
        np.asarray(mask, dtype=np.uint8),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(
    paths: list[Path],
    include_absolute_paths: bool = False,
) -> list[dict[str, object]]:
    """Record stable source identity without leaking local directory structure."""
    result: list[dict[str, object]] = []
    for index, path in enumerate(paths):
        entry: dict[str, object] = {
            "source_index": index,
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        if include_absolute_paths:
            entry["absolute_path"] = str(path)
        result.append(entry)
    return result



def build_source_provenance(
    paths: Iterable[str | Path],
    metadata_path: str | Path | None = None,
    release_mode: bool = False,
) -> tuple[list[dict[str, object]], str, int | None]:
    """Build a stable source/license manifest and optionally enforce release gates.

    Metadata JSON format:
      {"format":"qmini-source-provenance-v1","sources":[
        {"path":"corpus.jsonl","source_name":"...","source_uri":"https://...",
         "license_spdx":"CC-BY-4.0","collected_at":"2026-11-10",
         "language":"en","transformation_version":"raw-v1"}]}
    Paths are matched by basename. Duplicate basenames are rejected to avoid
    ambiguous provenance. Research mode records unknown provenance explicitly;
    release mode rejects unknown fields and scans records for likely PII/secrets.
    """
    source_paths = expand_data_paths(paths)
    if not source_paths:
        raise FileNotFoundError("No supported dataset files were found.")
    names = [path.name for path in source_paths]
    if len(set(names)) != len(names):
        raise ValueError(
            "Dataset files have duplicate basenames; use unique filenames so source provenance is unambiguous."
        )

    supplied: dict[str, dict[str, object]] = {}
    if metadata_path is not None:
        metadata_file = Path(metadata_path).expanduser()
        if not metadata_file.is_file():
            raise FileNotFoundError(f"Source metadata file not found: {metadata_file}")
        try:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Source metadata must be valid UTF-8 JSON.") from exc
        if not isinstance(payload, dict) or payload.get("format") != "qmini-source-provenance-v1":
            raise ValueError("Unsupported source metadata format; expected qmini-source-provenance-v1.")
        records = payload.get("sources")
        if not isinstance(records, list):
            raise ValueError("Source metadata must contain a 'sources' array.")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each source metadata entry must be a JSON object.")
            name = record.get("path")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each source metadata entry requires a non-empty 'path' basename.")
            name = Path(name).name
            if name in supplied:
                raise ValueError(f"Duplicate source metadata entry for {name!r}.")
            supplied[name] = record
        extras = sorted(set(supplied) - set(names))
        if extras:
            raise ValueError(
                "Source metadata references files that are not in this dataset input: "
                + ", ".join(extras)
            )
    elif release_mode:
        raise ValueError("Release mode requires --source-metadata with explicit source licenses.")

    base_files = _file_manifest(source_paths)
    result: list[dict[str, object]] = []
    for base in base_files:
        name = str(base["path"])
        record = supplied.get(name, {})
        fields = {
            "source_name": record.get("source_name") or name,
            "source_uri": record.get("source_uri"),
            "license_spdx": record.get("license_spdx") or "UNKNOWN",
            "collected_at": record.get("collected_at"),
            "language": record.get("language") or "und",
            "transformation_version": record.get("transformation_version") or "unknown",
        }
        if release_mode:
            required_fields = (
                "source_name", "source_uri", "license_spdx",
                "collected_at", "language", "transformation_version",
            )
            missing = [
                key for key in required_fields
                if not isinstance(record.get(key), str) or not record[key].strip()
            ]
            if missing:
                raise ValueError(
                    f"Release provenance for {name!r} is missing: {', '.join(missing)}."
                )
            license_value = str(fields["license_spdx"]).strip()
            if (
                license_value.casefold() in {"unknown", "unlicensed", "none", "n/a"}
                or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9.+-]*(?:\s+(?:AND|OR|WITH)\s+[A-Za-z0-9][A-Za-z0-9.+-]*)*",
                    license_value,
                )
            ):
                raise ValueError(
                    f"Release dataset rejects an unknown or unapproved license "
                    f"(or malformed SPDX expression) for {name!r}."
                )
            uri = urlsplit(str(fields["source_uri"]))
            if uri.scheme not in {"http", "https"} or not uri.netloc or uri.username or uri.password:
                raise ValueError(f"Release provenance for {name!r} needs a public HTTP(S) source URI.")
            try:
                date.fromisoformat(str(fields["collected_at"]))
            except ValueError as exc:
                raise ValueError(
                    f"Release provenance for {name!r} needs an ISO-8601 collection date."
                ) from exc
            language = str(fields["language"])
            if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language):
                raise ValueError(f"Release provenance for {name!r} needs a BCP-47 language tag.")
            if not str(fields["transformation_version"]).strip():
                raise ValueError(f"Release provenance for {name!r} needs a transformation version.")
        result.append({**base, **fields, "document_count": None})

    sensitive_records: int | None = None
    if release_mode:
        email_pattern = re.compile(
            r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9_-])"
        )
        secret_patterns = (
            re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
            re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
            re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
        )
        sensitive_records = 0
        for path in source_paths:
            for record in iter_source_records([path]):
                text = extract_text(record)
                if email_pattern.search(text) or any(
                    pattern.search(text) for pattern in secret_patterns
                ):
                    sensitive_records += 1
        if sensitive_records:
            raise ValueError(
                f"Release dataset blocked: {sensitive_records} records contain potential PII/secrets. "
                "No raw content was logged; remove or quarantine flagged records before retrying."
            )

    canonical = json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    provenance_sha256 = hashlib.sha256(canonical).hexdigest()
    return result, provenance_sha256, sensitive_records


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_token_streams(
    data_paths: Iterable[str | Path],
    tokenizer_path: str | Path,
    output_dir: str | Path,
    valid_fraction: float = 0.02,
    loss_mask_mode: str = "auto",
    include_absolute_source_paths: bool = False,
    near_duplicate_distance: int = 0,
    source_metadata_path: str | Path | None = None,
    release_mode: bool = False,
) -> dict[str, object]:
    """Prepare deduplicated train/validation token and assistant-loss-mask streams.

    loss_mask_mode:
      auto      - general text trains all tokens; chat/Q&A records train assistant outputs.
      all       - ordinary next-token pretraining over every token.
      assistant - only structured assistant responses contribute to loss.
    """
    from .tokenizer import load_tokenizer, special_token_id

    if not 0.0 < valid_fraction < 0.5:
        raise ValueError("valid_fraction must be greater than 0 and below 0.5.")
    if loss_mask_mode not in {"auto", "all", "assistant"}:
        raise ValueError("loss_mask_mode must be one of: auto, all, assistant.")
    _validate_near_duplicate_distance(near_duplicate_distance)

    paths = expand_data_paths(data_paths)
    if not paths:
        raise FileNotFoundError("No supported dataset files were found.")
    provenance_files, source_provenance_sha256, sensitive_records_screened = (
        build_source_provenance(
            paths, metadata_path=source_metadata_path, release_mode=release_mode
        )
    )

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    train_path = target / "train.bin"
    valid_path = target / "valid.bin"
    train_mask_path = target / "train.mask.bin"
    valid_mask_path = target / "valid.mask.bin"
    manifest_path = target / "dataset.json"
    output_paths = (train_path, valid_path, train_mask_path, valid_mask_path, manifest_path)
    if any(path.exists() for path in output_paths):
        raise FileExistsError(
            f"Existing dataset artifacts in {target} are preserved; choose a new output directory."
        )

    tokenizer_path_obj = Path(tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path_obj)
    tokenizer_sidecar = Path(str(tokenizer_path_obj) + ".qmini.json")
    if release_mode and not tokenizer_sidecar.is_file():
        raise ValueError(
            "Release preparation requires the matching tokenizer provenance sidecar; "
            "retrain the tokenizer with --release-mode and the same --source-metadata."
        )
    if tokenizer_sidecar.is_file():
        try:
            sidecar = json.loads(tokenizer_sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read tokenizer split manifest: {tokenizer_sidecar}"
            ) from exc
        if sidecar.get("format") != "qmini-tokenizer-manifest-v1":
            raise ValueError("Unsupported tokenizer split manifest format.")
        if sidecar.get("validation_fraction") != float(valid_fraction):
            raise ValueError(
                "Tokenizer and prepared dataset use different validation fractions. "
                "Rebuild them with the same --valid-fraction."
            )
        if sidecar.get("near_duplicate_hamming_distance") != near_duplicate_distance:
            raise ValueError(
                "Tokenizer and prepared dataset use different near-duplicate thresholds. "
                "Rebuild them with the same --near-duplicate-hamming-distance."
            )
        if sidecar.get("tokenizer_sha256") != _sha256_file(tokenizer_path_obj):
            raise ValueError("Tokenizer file does not match its Q-Mini split manifest.")
        if release_mode and sidecar.get("release_mode") is not True:
            raise ValueError(
                "Release preparation requires a tokenizer fitted with --release-mode "
                "so source licenses and PII/secrets were checked before BPE training."
            )
        saved_provenance = sidecar.get("source_provenance_sha256")
        provenance_enforced = release_mode or source_metadata_path is not None
        if provenance_enforced and saved_provenance != source_provenance_sha256:
            raise ValueError(
                "Tokenizer source provenance differs from prepared data. Rebuild the tokenizer "
                "with the same source metadata and source files."
            )
    bos_id = special_token_id(tokenizer, "<bos>")
    eos_id = special_token_id(tokenizer, "<eos>")
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 65535:
        raise ValueError("uint16 token streams support vocabulary sizes up to 65,535.")

    source_manifest_before = _file_manifest(paths, include_absolute_source_paths)
    fd, database_name = tempfile.mkstemp(
        prefix=".qmini_dedup_", suffix=".sqlite3", dir=target
    )
    os.close(fd)
    database_path = Path(database_name)
    temp_outputs = [
        target / "train.bin.tmp",
        target / "valid.bin.tmp",
        target / "train.mask.bin.tmp",
        target / "valid.mask.bin.tmp",
    ]
    stats: dict[str, object] = {
        "unique_documents": 0,
        "duplicate_documents": 0,
        "skipped_short_documents": 0,
        "skipped_without_targets": 0,
        "train_tokens": 0,
        "valid_tokens": 0,
        "train_target_tokens": 0,
        "valid_target_tokens": 0,
        "train_documents": 0,
        "valid_documents": 0,
        "near_duplicate_groups": 0,
        "near_duplicate_documents_grouped": 0,
        "near_duplicate_hamming_distance": near_duplicate_distance,
    }

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        _create_document_group_index(connection, near_duplicate_distance)
        connection.execute("CREATE TABLE written (digest BLOB PRIMARY KEY)")

        # First pass: exact duplicates and optional near-duplicate groups are
        # indexed on disk, never retained as a Python list in memory.
        for source in provenance_files:
            source["document_count"] = 0
        with connection:
            for source_index, source_path in enumerate(paths):
                for record in iter_source_records([source_path]):
                    normalized = extract_text(record).strip()
                    if len(normalized) < 2:
                        stats["skipped_short_documents"] += 1
                        continue
                    provenance_files[source_index]["document_count"] += 1
                    _digest, inserted = _register_document(
                        connection, normalized, near_duplicate_distance
                    )
                    if inserted:
                        stats["unique_documents"] += 1
                    else:
                        stats["duplicate_documents"] += 1

        unique_count = int(stats["unique_documents"])
        if unique_count < 2:
            raise ValueError(
                "At least two unique documents are required for non-empty train and validation sets."
            )

        grouping = _finalize_document_group_splits(connection, valid_fraction)
        stats["near_duplicate_groups"] = grouping["groups"]
        stats["near_duplicate_documents_grouped"] = (
            unique_count - grouping["groups"]
        )

        # Second pass: each unique source record is encoded once and split atomically.
        with (
            temp_outputs[0].open("wb") as train_file,
            temp_outputs[1].open("wb") as valid_file,
            temp_outputs[2].open("wb") as train_mask_file,
            temp_outputs[3].open("wb") as valid_mask_file,
        ):
            for source_path in paths:
                for record in iter_source_records([source_path]):
                    normalized = extract_text(record).strip()
                    if len(normalized) < 2:
                        continue
                    digest = hashlib.sha256(
                        _encode_dataset_text(normalized)
                    ).digest()
                    if connection.execute(
                        "SELECT 1 FROM documents WHERE digest=?", (digest,)
                    ).fetchone() is None:
                        raise RuntimeError("Source data changed during token preparation.")
                    inserted = connection.execute(
                        "INSERT OR IGNORE INTO written(digest) VALUES (?)", (digest,)
                    )
                    if inserted.rowcount == 0:
                        continue
                    is_validation = _document_is_validation(connection, digest)

                    encoded = _encode_record(
                        record, tokenizer, bos_id, eos_id, loss_mask_mode
                    )
                    if encoded is None:
                        stats["skipped_without_targets"] += 1
                        continue
                    token_ids, loss_mask = encoded
                    if token_ids.size and int(token_ids.max()) >= vocab_size:
                        raise ValueError("Tokenizer emitted an ID outside its declared vocabulary.")
                    if token_ids.size != loss_mask.size:
                        raise RuntimeError("Token and loss-mask lengths do not match.")
                    if not np.isin(loss_mask, (0, 1)).all():
                        raise RuntimeError("Loss mask must contain only binary values.")

                    if is_validation:
                        token_ids.tofile(valid_file)
                        loss_mask.tofile(valid_mask_file)
                        stats["valid_tokens"] += int(token_ids.size)
                        stats["valid_target_tokens"] += int(loss_mask.sum())
                        stats["valid_documents"] += 1
                    else:
                        token_ids.tofile(train_file)
                        loss_mask.tofile(train_mask_file)
                        stats["train_tokens"] += int(token_ids.size)
                        stats["train_target_tokens"] += int(loss_mask.sum())
                        stats["train_documents"] += 1

            for handle in (train_file, valid_file, train_mask_file, valid_mask_file):
                handle.flush()
                os.fsync(handle.fileno())

        source_manifest_after = _file_manifest(paths, include_absolute_source_paths)
        if source_manifest_after != source_manifest_before:
            raise RuntimeError("Source files changed while token streams were being prepared.")
        if int(stats["train_tokens"]) == 0 or int(stats["valid_tokens"]) == 0:
            raise RuntimeError("Token preparation produced an empty train or validation stream.")
        if int(stats["train_target_tokens"]) == 0 or int(stats["valid_target_tokens"]) == 0:
            raise RuntimeError(
                "At least one split has no supervised target tokens. Add assistant answers "
                "or use --loss-mask all for ordinary next-token pretraining."
            )

        for temporary, final in zip(temp_outputs, (
            train_path, valid_path, train_mask_path, valid_mask_path
        )):
            os.replace(temporary, final)

        corpus_descriptor = json.dumps(
            source_manifest_before, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        metadata: dict[str, object] = {
            "format": "qmini-token-stream-v3",
            **stats,
            "vocab_size": int(vocab_size),
            "valid_fraction_requested": float(valid_fraction),
            "dtype": "uint16",
            "loss_mask_dtype": "uint8",
            "loss_mask_mode": loss_mask_mode,
            "deduplication": (
                "exact-document-sha256"
                if near_duplicate_distance == 0
                else "exact-document-sha256+simhash64"
            ),
            "near_duplicate_hamming_distance": near_duplicate_distance,
            "absolute_source_paths_included": bool(include_absolute_source_paths),
            "source_files": [
                {
                    **entry,
                    "source_name": provenance["source_name"],
                    "source_uri": provenance["source_uri"],
                    "license_spdx": provenance["license_spdx"],
                    "collected_at": provenance["collected_at"],
                    "language": provenance["language"],
                    "transformation_version": provenance["transformation_version"],
                    "document_count": provenance["document_count"],
                }
                for entry, provenance in zip(source_manifest_before, provenance_files)
            ],
            "source_provenance_sha256": source_provenance_sha256,
            "source_provenance_enforced": bool(release_mode or source_metadata_path is not None),
            "release_mode": bool(release_mode),
            "sensitive_records_screened": sensitive_records_screened,
            "corpus_sha256": hashlib.sha256(corpus_descriptor).hexdigest(),
            "tokenizer_sha256": _sha256_file(tokenizer_path_obj),
            "train_tokens_sha256": _sha256_file(train_path),
            "valid_tokens_sha256": _sha256_file(valid_path),
            "train_mask_sha256": _sha256_file(train_mask_path),
            "valid_mask_sha256": _sha256_file(valid_mask_path),
        }
        _write_json_atomic(manifest_path, metadata)
        return metadata

    except Exception:
        for path in (*temp_outputs, *output_paths):
            path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
        Path(str(database_path) + "-journal").unlink(missing_ok=True)
        Path(str(database_path) + "-wal").unlink(missing_ok=True)
        Path(str(database_path) + "-shm").unlink(missing_ok=True)
