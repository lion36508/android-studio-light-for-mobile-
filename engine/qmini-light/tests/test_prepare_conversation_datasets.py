"""Tests for scripts/prepare_conversation_datasets.py (offline, no network)."""

import hashlib
import importlib.util
import zipfile
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_conversation_datasets.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_conversation_datasets", _SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = _load_script()


def test_dailydialog_turns_alternate_user_assistant():
    raw = (
        "Hello there.__eou__Hi! How are you?__eou__I am fine, thanks.__eou__Great to hear.\n"
        "Good morning.__eou__Morning!\n"
    )
    records, dropped = prep.convert_dailydialog_text(raw)
    assert len(records) == 2
    assert dropped == {"empty": 0, "no_assistant": 0}
    assert records[0]["messages"] == [
        {"role": "user", "content": "Hello there."},
        {"role": "assistant", "content": "Hi! How are you?"},
        {"role": "user", "content": "I am fine, thanks."},
        {"role": "assistant", "content": "Great to hear."},
    ]
    assert records[1]["messages"] == [
        {"role": "user", "content": "Good morning."},
        {"role": "assistant", "content": "Morning!"},
    ]


def test_dailydialog_drops_empty_turns_and_single_turn_dialogues():
    raw = (
        "Only one turn.\n"
        "Hi.__eou__  __eou__Hello again.__eou__Hi there.\n"
        "__eou__\n"
    )
    records, dropped = prep.convert_dailydialog_text(raw)
    assert len(records) == 1
    assert dropped == {"empty": 1, "no_assistant": 1}
    assert records[0]["messages"] == [
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello again."},
        {"role": "user", "content": "Hi there."},
    ]


def test_converted_records_use_only_user_and_assistant_roles():
    raw = "A.__eou__B.__eou__C.__eou__D.\n"
    records, _dropped = prep.convert_dailydialog_text(raw)
    roles = {message["role"] for record in records for message in record["messages"]}
    assert roles <= {"user", "assistant"}


def _oasst1_row(
    message_id,
    parent_id,
    text,
    role="prompter",
    lang="en",
    deleted=False,
    tree="tree-1",
    review_result=True,
    synthetic=False,
    rank=0,
    tree_state="ready_for_export",
):
    return {
        "message_id": message_id,
        "parent_id": parent_id,
        "text": text,
        "role": role,
        "lang": lang,
        "review_result": review_result,
        "deleted": deleted,
        "rank": rank,
        "synthetic": synthetic,
        "message_tree_id": tree,
        "tree_state": tree_state,
    }


def test_oasst1_flattens_first_child_chain_depth_first():
    rows = [
        _oasst1_row("m1", None, "What is 2+2?", "prompter"),
        _oasst1_row("m2", "m1", "It is 4.", "assistant"),
        _oasst1_row("m3", "m2", "Thanks!", "prompter"),
        _oasst1_row("m4", "m2", "Another reply.", "assistant"),
        _oasst1_row("m5", "m3", "You are welcome.", "assistant"),
    ]
    records, stats = prep.flatten_oasst1_rows(rows)
    assert len(records) == 1
    assert records[0]["messages"] == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It is 4."},
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "You are welcome."},
    ]
    assert stats["kept_messages"] == 5
    assert stats["trees"] == 1


def test_oasst1_filters_non_english_deleted_and_empty():
    rows = [
        _oasst1_row("m1", None, "Hello.", "prompter"),
        _oasst1_row("m2", "m1", "Bonjour.", "assistant", lang="fr"),
        _oasst1_row("m3", "m1", "Hi there.", "assistant"),
        _oasst1_row("m4", "m3", "   ", "prompter"),
        _oasst1_row("m5", "m3", "Deleted reply.", "assistant", deleted=True),
        _oasst1_row("m6", "m3", "How are you?", "prompter"),
    ]
    records, stats = prep.flatten_oasst1_rows(rows)
    assert len(records) == 1
    assert records[0]["messages"] == [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hi there."},
        {"role": "user", "content": "How are you?"},
    ]
    assert stats["dropped_messages"] == {
        "non_english": 1,
        "not_ready": 0,
        "unreviewed": 0,
        "deleted": 1,
        "synthetic": 0,
        "missing_rank": 0,
        "non_top_rank": 0,
        "empty": 1,
        "other_role": 0,
    }


def test_oasst1_excludes_synthetic_unreviewed_not_ready_and_non_top_ranked_rows():
    rows = [
        _oasst1_row("u1", None, "Question.", "prompter"),
        _oasst1_row("a-synthetic", "u1", "Generated answer.", "assistant", synthetic=True),
        _oasst1_row("a-unreviewed", "u1", "Unreviewed answer.", "assistant", review_result=False),
        _oasst1_row("a-not-ready", "u1", "Not exported.", "assistant", tree_state="halted_by_moderator"),
        _oasst1_row("a-ranked-1", "u1", "Second choice.", "assistant", rank=1),
        _oasst1_row("a-missing-rank", "u1", "Unranked answer.", "assistant", rank=None),
        _oasst1_row("a-best", "u1", "Reviewed human answer.", "assistant", rank=0),
        _oasst1_row("u2", "a-best", "Thanks.", "prompter"),
        _oasst1_row("a2", "u2", "You are welcome.", "assistant", rank=0),
    ]
    records, stats = prep.flatten_oasst1_rows(rows)

    assert len(records) == 1
    assert records[0]["messages"] == [
        {"role": "user", "content": "Question."},
        {"role": "assistant", "content": "Reviewed human answer."},
        {"role": "user", "content": "Thanks."},
        {"role": "assistant", "content": "You are welcome."},
    ]
    assert stats["dropped_messages"]["synthetic"] == 1
    assert stats["dropped_messages"]["unreviewed"] == 1
    assert stats["dropped_messages"]["not_ready"] == 1
    assert stats["dropped_messages"]["non_top_rank"] == 1
    assert stats["dropped_messages"]["missing_rank"] == 1


def test_oasst1_merges_consecutive_same_role_turns():
    rows = [
        _oasst1_row("m1", None, "First part.", "prompter"),
        _oasst1_row("m2", "m1", "Second part.", "prompter"),
        _oasst1_row("m3", "m2", "Answer.", "assistant"),
    ]
    records, _stats = prep.flatten_oasst1_rows(rows)
    assert records[0]["messages"] == [
        {"role": "user", "content": "First part.\nSecond part."},
        {"role": "assistant", "content": "Answer."},
    ]


def test_oasst1_drops_trees_without_an_assistant_turn():
    rows = [
        _oasst1_row("m1", None, "Hello?", "prompter"),
        _oasst1_row("m2", None, "Hi.", "prompter", tree="tree-2"),
        _oasst1_row("m3", "m2", "Hello back.", "assistant", tree="tree-2"),
    ]
    records, stats = prep.flatten_oasst1_rows(rows)
    assert len(records) == 1
    assert records[0]["messages"][0]["content"] == "Hi."
    assert stats["dropped_trees"]["no_assistant"] == 1
    assert stats["trees"] == 2


def test_verify_file_accepts_matching_hash_and_size(tmp_path):
    path = tmp_path / "sample.bin"
    data = b"hello conversation"
    path.write_bytes(data)
    assert prep.verify_file(path, hashlib.sha256(data).hexdigest(), len(data)) is True


def test_verify_file_rejects_wrong_hash_size_and_missing(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"hello conversation")
    with pytest.raises(ValueError, match="SHA-256"):
        prep.verify_file(path, "0" * 64, len(b"hello conversation"))
    with pytest.raises(ValueError, match="size"):
        prep.verify_file(
            path, hashlib.sha256(b"hello conversation").hexdigest(), 999
        )
    with pytest.raises(FileNotFoundError):
        prep.verify_file(tmp_path / "missing.bin", "0" * 64, 0)


def test_manifest_round_trip_and_tamper_detection(tmp_path):
    records = [
        {
            "messages": [
                {"role": "user", "content": "Hi."},
                {"role": "assistant", "content": "Hello."},
            ]
        }
    ]
    out_path = tmp_path / "dailydialog_train.jsonl"
    prep.write_jsonl(out_path, records)
    manifest = prep.build_manifest(
        dataset="dailydialog",
        source={"repository": "roskoN/dailydialog"},
        output_files=[prep.describe_output_file(out_path, records)],
        counts={"train": {"dialogues": 1, "utterances": 2}},
    )
    manifest_path = tmp_path / "manifest.json"
    prep.write_manifest(manifest_path, manifest)
    assert prep.manifest_outputs_valid(manifest_path) is True
    out_path.write_text("tampered\n", encoding="utf-8")
    assert prep.manifest_outputs_valid(manifest_path) is False
    assert prep.manifest_outputs_valid(tmp_path / "missing.json") is False


def test_read_dailydialog_split_text_from_zip(tmp_path):
    zip_path = tmp_path / "train.zip"
    text = "Hello.__eou__Hi.\nGood day.__eou__Hello.\n"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("dialogues_train.txt", text)
        archive.writestr("dialogues_emotion_train.txt", "0 1\n0 1\n")
    assert prep.read_dailydialog_split_text(zip_path, "train") == text


def test_read_dailydialog_split_text_rejects_missing_member(tmp_path):
    zip_path = tmp_path / "train.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("something_else.txt", "x")
    with pytest.raises(ValueError, match="dialogues_train.txt"):
        prep.read_dailydialog_split_text(zip_path, "train")


def test_read_oasst1_train_rows_rejects_wrong_columns(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table({"message_id": ["m1"], "text": ["Hi."]})
    path = tmp_path / "train.parquet"
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="missing columns"):
        prep.read_oasst1_train_rows(path)


def test_read_oasst1_train_rows_rejects_wrong_row_count(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "message_id": ["m1", "m2"],
            "parent_id": [None, "m1"],
            "text": ["Hi.", "Hello."],
            "role": ["prompter", "assistant"],
            "lang": ["en", "en"],
            "review_result": [True, True],
            "deleted": [False, False],
            "rank": [None, 0],
            "synthetic": [False, False],
            "message_tree_id": ["t1", "t1"],
            "tree_state": ["ready_for_export", "ready_for_export"],
        }
    )
    path = tmp_path / "train.parquet"
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="rows"):
        prep.read_oasst1_train_rows(path)


def test_pinned_dataset_constants():
    assert len(prep.OASST1_TRAIN_PARQUET_SHA256) == 64
    assert {
        "review_result", "deleted", "rank", "synthetic", "tree_state"
    }.issubset(prep.OASST1_REQUIRED_COLUMNS)
    assert prep.OASST1_TRAIN_ROWS == 84437
    assert prep.DAILYDIALOG_SPLITS["train"]["dialogues"] == 11118
    assert prep.DAILYDIALOG_SPLITS["validation"]["dialogues"] == 1000
    assert prep.DAILYDIALOG_SPLITS["test"]["dialogues"] == 1000

def test_pinned_dailydialog_sha256_hashes():
    assert prep.DAILYDIALOG_SPLITS["train"]["sha256"] == "c5179ab5a9a86a77b9d29114087c6b82cc4cb366abea25a43a0c24be761133f7"
    assert prep.DAILYDIALOG_SPLITS["validation"]["sha256"] == "1dbcb00a0b4fe709788ce6604b561ba76557c88fe41f049d7899287838f0ec6a"
    assert prep.DAILYDIALOG_SPLITS["test"]["sha256"] == "cfffc0bab93a3697c2562a9ee35c4533f0de2a93e86a4e4d3ea45cbde4bea00f"
    for split in prep.DAILYDIALOG_SPLITS.values():
        assert len(split["sha256"]) == 64
        assert set(split["sha256"]) <= set("0123456789abcdef")
