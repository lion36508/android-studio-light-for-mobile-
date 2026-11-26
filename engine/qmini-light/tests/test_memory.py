import json

import pytest

from qmini.memory import LocalMemoryStore


def test_memory_persists_deduplicates_searches_and_forgets(tmp_path):
    database = tmp_path / "memory.sqlite3"
    store = LocalMemoryStore(database)

    first = store.remember(
        "My Q-Mini project uses causal convolution and sparse experts.",
        tags=["project", "architecture"],
    )
    duplicate = store.remember(
        "My Q-Mini project uses causal convolution and sparse experts.",
        tags=["Q-Mini"],
    )
    assert first["created"] is True
    assert duplicate["created"] is False
    assert duplicate["id"] == first["id"]
    assert set(duplicate["tags"]) == {"project", "architecture", "Q-Mini"}
    assert store.count() == 1

    reopened = LocalMemoryStore(database)
    results = reopened.search("causal convolution", limit=5)
    assert len(results) == 1
    assert results[0]["id"] == first["id"]
    assert "sparse experts" in results[0]["content"]

    assert reopened.forget(first["id"]) is True
    assert reopened.forget(first["id"]) is False
    assert reopened.count() == 0


def test_memory_export_is_json_and_empty_query_lists_recent(tmp_path):
    store = LocalMemoryStore(tmp_path / "memory.sqlite3")
    store.remember("First note about Python.", tags=["coding"])
    store.remember("Second note about physics.", tags=["science"])

    recent = store.search("", limit=10)
    assert len(recent) == 2

    export_path = store.export_json(tmp_path / "backup" / "memories.json")
    data = json.loads(export_path.read_text(encoding="utf-8"))
    assert data["format"] == "qmini-memory-export-v1"
    assert data["count"] == 2
    assert [item["content"] for item in data["records"]]


def test_memory_validates_inputs(tmp_path):
    store = LocalMemoryStore(tmp_path / "memory.sqlite3")
    with pytest.raises(ValueError, match="cannot be empty"):
        store.remember("   ")
    with pytest.raises(ValueError, match="positive integer"):
        store.forget(0)
    with pytest.raises(ValueError, match="limit"):
        store.search("python", limit=0)


def test_memory_is_explicit_no_implicit_records(tmp_path):
    store = LocalMemoryStore(tmp_path / "memory.sqlite3")
    assert store.count() == 0
