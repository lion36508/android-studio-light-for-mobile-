"""Prepare DailyDialog and OpenAssistant oasst1 (English only) conversation data.

This script is run by the Colab notebook (V11), not by a developer machine:
it downloads both datasets from Hugging Face at pinned revisions, verifies
every downloaded file (SHA-256, size, row counts and columns), converts the
dialogues to the pipeline's conversation format -- one JSON object per line,
``{"messages": [{"role": "user" | "assistant", "content": ...}]}`` -- and
writes them into Google Drive under ``MyDrive/QMini-Light-0.7M/datasets/``.

Outputs::

    <datasets-dir>/dailydialog/dailydialog_train.jsonl
    <datasets-dir>/dailydialog/dailydialog_validation.jsonl
    <datasets-dir>/dailydialog/dailydialog_test.jsonl
    <datasets-dir>/dailydialog/manifest.json
    <datasets-dir>/oasst1_en/oasst1_en_train.jsonl
    <datasets-dir>/oasst1_en/manifest.json

The script is idempotent: when the JSONL files and their manifest already
exist and verify against the manifest, that dataset is skipped. Pass
``--force`` to re-download and re-convert.

DailyDialog is licensed CC BY-NC-SA 4.0 (non-commercial). OpenAssistant
oasst1 is licensed Apache-2.0. Both licences are recorded in the manifests.

Only the standard library is needed for DailyDialog. Reading the OASST1
parquet file needs pyarrow (preinstalled in Colab); it is imported lazily
so preparing DailyDialog alone works without it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_FORMAT = "qmini-conversation-dataset-v1"
SCRIPT_VERSION = "1.0.0"

DEFAULT_DATASETS_DIR = "/content/drive/MyDrive/QMini-Light-0.7M/datasets"

# ---------------------------------------------------------------------------
# DailyDialog (roskoN/dailydialog, config "full")
# ---------------------------------------------------------------------------
DAILYDIALOG_REPO = "roskoN/dailydialog"
DAILYDIALOG_REVISION = "5214b2a66405abf87fd229e5c1007985501ffe3e"
DAILYDIALOG_LICENSE = "CC-BY-NC-SA-4.0"
DAILYDIALOG_SPLITS = {
    "train": {
        "file": "train.zip",
        "size_bytes": 1935190,
        "sha256": "c5179ab5a9a86a77b9d29114087c6b82cc4cb366abea25a43a0c24be761133f7",
        "dialogues": 11118,
    },
    "validation": {
        "file": "validation.zip",
        "size_bytes": 180450,
        "sha256": "1dbcb00a0b4fe709788ce6604b561ba76557c88fe41f049d7899287838f0ec6a",
        "dialogues": 1000,
    },
    "test": {
        "file": "test.zip",
        "size_bytes": 178601,
        "sha256": "cfffc0bab93a3697c2562a9ee35c4533f0de2a93e86a4e4d3ea45cbde4bea00f",
        "dialogues": 1000,
    },
}
DAILYDIALOG_UTTERANCE_SEPARATOR = "__eou__"

# ---------------------------------------------------------------------------
# OpenAssistant oasst1 (OpenAssistant/oasst1), English only, train split
# ---------------------------------------------------------------------------
OASST1_REPO = "OpenAssistant/oasst1"
OASST1_REVISION = "fdf72ae0827c1cda404ff25b6603abec9e3399b"
OASST1_LICENSE = "Apache-2.0"
OASST1_TRAIN_PARQUET = "data/train-00000-of-00001-b42a775f407cee45.parquet"
OASST1_TRAIN_PARQUET_SIZE_BYTES = 39516251
OASST1_TRAIN_PARQUET_SHA256 = "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e"
OASST1_TRAIN_ROWS = 84437
OASST1_REQUIRED_COLUMNS = (
    "message_id",
    "parent_id",
    "text",
    "role",
    "lang",
    "review_result",
    "deleted",
    "rank",
    "synthetic",
    "message_tree_id",
    "tree_state",
)


def sha256_file(path):
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, expected_sha256, expected_size_bytes):
    """Raise unless ``path`` exists with the expected size and SHA-256."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")
    size = path.stat().st_size
    if size != expected_size_bytes:
        raise ValueError(
            f"{path}: size {size} bytes, expected {expected_size_bytes} bytes"
        )
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {actual}, expected {expected_sha256}")
    return True


def hf_resolve_url(repo, revision, path):
    """Return the Hugging Face resolve URL for a file at a pinned revision."""
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def download_verified(url, dest, expected_sha256, expected_size_bytes):
    """Download ``url`` to ``dest`` (skipping it when already verified)."""
    dest = Path(dest)
    if dest.is_file():
        try:
            verify_file(dest, expected_sha256, expected_size_bytes)
            print(f"already downloaded and verified: {dest}")
            return dest
        except (ValueError, FileNotFoundError):
            print(f"re-downloading (cached copy failed verification): {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    verify_file(dest, expected_sha256, expected_size_bytes)
    print(f"downloaded and verified: {dest} ({expected_size_bytes} bytes)")
    return dest


def convert_dailydialog_text(raw_text):
    """Convert DailyDialog raw text to conversation records.

    Each line is one dialogue; utterances are separated by ``__eou__``.
    Turns alternate speakers: turn 0 is the user, turn 1 the assistant,
    and so on. Empty turns are dropped. A dialogue that ends up without
    an assistant turn (for example a single-turn dialogue) is dropped,
    because the pipeline's assistant-only loss mask skips records with
    no assistant content.

    Returns ``(records, dropped)`` where ``dropped`` counts the dialogues
    dropped as ``empty`` or ``no_assistant``.
    """
    records = []
    dropped = {"empty": 0, "no_assistant": 0}
    for line in raw_text.splitlines():
        utterances = [
            part.strip()
            for part in line.split(DAILYDIALOG_UTTERANCE_SEPARATOR)
        ]
        utterances = [utterance for utterance in utterances if utterance]
        if not utterances:
            dropped["empty"] += 1
            continue
        messages = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": utterance,
            }
            for index, utterance in enumerate(utterances)
        ]
        if not any(message["role"] == "assistant" for message in messages):
            dropped["no_assistant"] += 1
            continue
        records.append({"messages": messages})
    return records, dropped


def read_dailydialog_split_text(zip_path, split):
    """Return the raw dialogues text for one DailyDialog split."""
    wanted = f"dialogues_{split}.txt"
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        matches = [name for name in names if Path(name).name == wanted]
        if len(matches) != 1:
            raise ValueError(
                f"{zip_path}: expected exactly one {wanted} in the archive, "
                f"found {matches} (archive contains: {names})"
            )
        return archive.read(matches[0]).decode("utf-8")


def flatten_oasst1_rows(rows):
    """Flatten only reviewed, non-synthetic English OASST1 messages.

    The input schema must include review_result, deleted, synthetic, rank and
    tree_state. Only ready-for-export, positively reviewed, non-deleted,
    explicitly non-synthetic English messages are kept. For assistant replies,
    only rank 0 is retained. Unknown/missing safety flags fail closed.
    Conversation trees remain intact for downstream document-level splitting.
    """
    kept = []
    dropped = {
        "non_english": 0,
        "not_ready": 0,
        "unreviewed": 0,
        "deleted": 0,
        "synthetic": 0,
        "missing_rank": 0,
        "non_top_rank": 0,
        "empty": 0,
        "other_role": 0,
    }
    for row in rows:
        if row.get("lang") != "en":
            dropped["non_english"] += 1
            continue
        if row.get("tree_state") != "ready_for_export":
            dropped["not_ready"] += 1
            continue
        if row.get("review_result") is not True:
            dropped["unreviewed"] += 1
            continue
        if row.get("deleted") is not False:
            dropped["deleted"] += 1
            continue
        if row.get("synthetic") is not False:
            dropped["synthetic"] += 1
            continue
        role = row.get("role")
        if role == "prompter":
            mapped = "user"
        elif role == "assistant":
            rank = row.get("rank")
            if rank is None:
                dropped["missing_rank"] += 1
                continue
            try:
                rank = int(rank)
            except (TypeError, ValueError):
                dropped["non_top_rank"] += 1
                continue
            if rank != 0:
                dropped["non_top_rank"] += 1
                continue
            mapped = "assistant"
        else:
            dropped["other_role"] += 1
            continue
        text = (row.get("text") or "").strip()
        if not text:
            dropped["empty"] += 1
            continue
        kept.append(
            {
                "message_id": row.get("message_id"),
                "parent_id": row.get("parent_id"),
                "message_tree_id": row.get("message_tree_id"),
                "role": mapped,
                "content": text,
            }
        )
    trees = {}
    order = []
    for message in kept:
        tree_id = message["message_tree_id"]
        if tree_id not in trees:
            trees[tree_id] = []
            order.append(tree_id)
        trees[tree_id].append(message)
    records = []
    dropped_trees = {"no_root": 0, "no_user_start": 0, "no_assistant": 0}
    for tree_id in order:
        messages = trees[tree_id]
        children = {}
        root = None
        for message in messages:
            parent = message["parent_id"]
            if parent is None:
                root = message
            else:
                children.setdefault(parent, []).append(message)
        if root is None:
            dropped_trees["no_root"] += 1
            continue
        chain = [root]
        seen = {root["message_id"]}
        while chain[-1]["message_id"] in children:
            nxt = children[chain[-1]["message_id"]][0]
            if nxt["message_id"] in seen:
                break
            seen.add(nxt["message_id"])
            chain.append(nxt)
        dialogue = []
        for message in chain:
            if dialogue and dialogue[-1]["role"] == message["role"]:
                dialogue[-1]["content"] += "\n" + message["content"]
            else:
                dialogue.append(
                    {"role": message["role"], "content": message["content"]}
                )
        if not dialogue or dialogue[0]["role"] != "user":
            dropped_trees["no_user_start"] += 1
            continue
        if not any(turn["role"] == "assistant" for turn in dialogue):
            dropped_trees["no_assistant"] += 1
            continue
        records.append({"messages": dialogue})
    stats = {
        "kept_messages": len(kept),
        "trees": len(order),
        "dropped_messages": dropped,
        "dropped_trees": dropped_trees,
    }
    return records, stats


def read_oasst1_train_rows(parquet_path):
    """Read the OASST1 train parquet file and return a list of row dicts."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    columns = set(table.column_names)
    missing = [name for name in OASST1_REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"{parquet_path}: missing columns: {missing}")
    if table.num_rows != OASST1_TRAIN_ROWS:
        raise ValueError(
            f"{parquet_path}: {table.num_rows} rows, "
            f"expected {OASST1_TRAIN_ROWS}"
        )
    return table.to_pylist()


def write_jsonl(path, records):
    """Write conversation records as one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def count_utterances(records):
    """Return the total number of turns across conversation records."""
    return sum(len(record["messages"]) for record in records)


def describe_output_file(path, records):
    """Describe a written JSONL file for a manifest (name, hash, counts)."""
    path = Path(path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "dialogues": len(records),
        "utterances": count_utterances(records),
    }


def build_manifest(dataset, source, output_files, counts):
    """Build a manifest dict recording provenance and output checksums."""
    return {
        "format": MANIFEST_FORMAT,
        "script_version": SCRIPT_VERSION,
        "dataset": dataset,
        "source": source,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "files": output_files,
        "counts": counts,
    }


def write_manifest(path, manifest):
    """Write a manifest dict as indented, key-sorted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def manifest_outputs_valid(manifest_path):
    """Return True when every file in a manifest exists and verifies."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    if manifest.get("format") != MANIFEST_FORMAT:
        return False
    for entry in manifest.get("files", []):
        path = manifest_path.parent / entry["path"]
        if not path.is_file():
            return False
        if path.stat().st_size != entry["size_bytes"]:
            return False
        if sha256_file(path) != entry["sha256"]:
            return False
    return True


def prepare_dailydialog(datasets_dir, cache_dir, force=False):
    """Download, verify and convert DailyDialog; return the manifest path."""
    out_dir = Path(datasets_dir) / "dailydialog"
    manifest_path = out_dir / "manifest.json"
    if not force and manifest_outputs_valid(manifest_path):
        print("DailyDialog already prepared and verified, skipping.")
        return manifest_path
    cache_dir = Path(cache_dir) / "dailydialog"
    all_records = {}
    source_files = []
    for split, spec in DAILYDIALOG_SPLITS.items():
        url = hf_resolve_url(DAILYDIALOG_REPO, DAILYDIALOG_REVISION, spec["file"])
        zip_path = download_verified(
            url, cache_dir / spec["file"], spec["sha256"], spec["size_bytes"]
        )
        raw_text = read_dailydialog_split_text(zip_path, split)
        records, dropped = convert_dailydialog_text(raw_text)
        total = len(records) + dropped["empty"] + dropped["no_assistant"]
        if total != spec["dialogues"]:
            raise ValueError(
                f"DailyDialog {split}: {total} dialogues, "
                f"expected {spec['dialogues']}"
            )
        out_path = out_dir / f"dailydialog_{split}.jsonl"
        write_jsonl(out_path, records)
        all_records[split] = records
        source_files.append(
            {
                "path": spec["file"],
                "sha256": spec["sha256"],
                "size_bytes": spec["size_bytes"],
                "dialogues": spec["dialogues"],
            }
        )
        print(
            f"DailyDialog {split}: {len(records)} dialogues, "
            f"{count_utterances(records)} utterances (dropped: {dropped})"
        )
    files = [
        describe_output_file(out_dir / f"dailydialog_{split}.jsonl", records)
        for split, records in all_records.items()
    ]
    counts = {
        split: {
            "dialogues": len(records),
            "utterances": count_utterances(records),
        }
        for split, records in all_records.items()
    }
    manifest = build_manifest(
        dataset="dailydialog",
        source={
            "repository": DAILYDIALOG_REPO,
            "revision": DAILYDIALOG_REVISION,
            "config": "full",
            "license": DAILYDIALOG_LICENSE,
            "url": f"https://huggingface.co/datasets/{DAILYDIALOG_REPO}",
            "files": source_files,
        },
        output_files=files,
        counts=counts,
    )
    write_manifest(manifest_path, manifest)
    print(f"DailyDialog manifest written: {manifest_path}")
    return manifest_path


def prepare_oasst1(datasets_dir, cache_dir, force=False):
    """Download, verify and convert OASST1 English train; return manifest path."""
    out_dir = Path(datasets_dir) / "oasst1_en"
    manifest_path = out_dir / "manifest.json"
    if not force and manifest_outputs_valid(manifest_path):
        print("OASST1 (English) already prepared and verified, skipping.")
        return manifest_path
    cache_dir = Path(cache_dir) / "oasst1"
    url = hf_resolve_url(OASST1_REPO, OASST1_REVISION, OASST1_TRAIN_PARQUET)
    parquet_path = download_verified(
        url,
        cache_dir / Path(OASST1_TRAIN_PARQUET).name,
        OASST1_TRAIN_PARQUET_SHA256,
        OASST1_TRAIN_PARQUET_SIZE_BYTES,
    )
    rows = read_oasst1_train_rows(parquet_path)
    records, stats = flatten_oasst1_rows(rows)
    print(
        f"OASST1 train: {len(rows)} rows read; "
        f"{stats['kept_messages']} English messages kept; "
        f"dropped messages: {stats['dropped_messages']}; "
        f"dropped trees: {stats['dropped_trees']}; "
        f"{len(records)} dialogues, {count_utterances(records)} utterances"
    )
    out_path = out_dir / "oasst1_en_train.jsonl"
    write_jsonl(out_path, records)
    manifest = build_manifest(
        dataset="oasst1_en",
        source={
            "repository": OASST1_REPO,
            "revision": OASST1_REVISION,
            "license": OASST1_LICENSE,
            "url": f"https://huggingface.co/datasets/{OASST1_REPO}",
            "files": [
                {
                    "path": OASST1_TRAIN_PARQUET,
                    "sha256": OASST1_TRAIN_PARQUET_SHA256,
                    "size_bytes": OASST1_TRAIN_PARQUET_SIZE_BYTES,
                    "rows": OASST1_TRAIN_ROWS,
                }
            ],
        },
        output_files=[describe_output_file(out_path, records)],
        counts={
            "train": {
                "dialogues": len(records),
                "utterances": count_utterances(records),
            },
            "rows_read": len(rows),
            "english_train_messages_kept": stats["kept_messages"],
            "dropped_messages": stats["dropped_messages"],
            "dropped_trees": stats["dropped_trees"],
        },
    )
    write_manifest(manifest_path, manifest)
    print(f"OASST1 manifest written: {manifest_path}")
    return manifest_path


def main(argv=None):
    """Prepare both conversation datasets into ``--datasets-dir``."""
    parser = argparse.ArgumentParser(
        description="Prepare DailyDialog and OASST1 (English only) "
        "conversation data for the Q-Mini pipeline."
    )
    parser.add_argument(
        "--datasets-dir",
        default=DEFAULT_DATASETS_DIR,
        help="Drive datasets directory (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="download cache directory (default: a temporary directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download and re-convert even when verified outputs exist",
    )
    parser.add_argument(
        "--only",
        choices=("dailydialog", "oasst1"),
        default=None,
        help="prepare only one dataset",
    )
    args = parser.parse_args(argv)
    datasets_dir = Path(args.datasets_dir)
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = Path(tempfile.mkdtemp(prefix="qmini-conversation-cache-"))
    if args.only in (None, "dailydialog"):
        prepare_dailydialog(datasets_dir, cache_dir, force=args.force)
    if args.only in (None, "oasst1"):
        prepare_oasst1(datasets_dir, cache_dir, force=args.force)
    print("Conversation datasets prepared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
