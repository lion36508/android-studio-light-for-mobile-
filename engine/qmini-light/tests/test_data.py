import json
from pathlib import Path

import numpy as np
import pytest

from qmini.data import _encode_dataset_text, _encode_record, _record_segments, extract_text, iter_text_records, prepare_token_streams
from qmini.tokenizer import special_token_id, train_tokenizer


def test_dataset_text_rejects_unpaired_unicode_surrogates():
    with pytest.raises(ValueError, match="invalid Unicode surrogate"):
        _encode_dataset_text("malformed " + chr(0xD800))


def test_extract_chat_messages():
    text = extract_text({"messages": [
        {"role": "user", "content": "Explain gravity."},
        {"role": "assistant", "content": "Gravity attracts masses."},
    ]})
    assert "[USER]" in text and "Explain gravity." in text
    assert "Gravity attracts masses." in text


def test_plain_text_reader_rejects_malformed_utf8_instead_of_replacing_bytes(tmp_path):
    from qmini.data import iter_source_records

    source = tmp_path / "malformed.txt"
    source.write_bytes(b"valid first line\ninvalid byte: \xff\n")

    with pytest.raises(ValueError, match="Invalid UTF-8.*malformed.txt"):
        list(iter_source_records([source]))


def test_plain_text_reader_rejects_invalid_chunk_size(tmp_path):
    from qmini.data import _iter_plain

    source = tmp_path / "plain.txt"
    source.write_text("valid text", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_chars must be positive"):
        list(_iter_plain(source, chunk_chars=0))


def test_jsonl_and_plain_text_readers(tmp_path):
    jsonl = tmp_path / "chat.jsonl"
    jsonl.write_text(
        json.dumps({"question": "Q?", "answer": "A!"}) + "\n",
        encoding="utf-8",
    )
    plain = tmp_path / "notes.txt"
    plain.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
    records = list(iter_text_records([jsonl, plain]))
    assert any("Q?" in item and "A!" in item for item in records)
    assert any("First paragraph." in item for item in records)
    assert any("Second paragraph." in item for item in records)


def test_prepare_deduplicates_and_guarantees_nonempty_splits(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    records = [
        {"text": "The first distinct document has enough text."},
        {"text": "The second distinct document also has enough text."},
        {"text": "The first distinct document has enough text."},
    ]
    corpus.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer(
        [corpus], tokenizer_path, vocab_size=300, min_frequency=1,
        validation_fraction=0.01,
    )

    output = tmp_path / "tokens"
    report = prepare_token_streams(
        [corpus], tokenizer_path, output, valid_fraction=0.01
    )
    assert report["unique_documents"] == 2
    assert report["duplicate_documents"] == 1
    assert report["train_documents"] == 1
    assert report["valid_documents"] == 1
    assert report["train_tokens"] > 0
    assert report["valid_tokens"] > 0
    assert report["format"] == "qmini-token-stream-v3"
    assert (output / "dataset.json").is_file()

    for split in ("train", "valid"):
        tokens = np.memmap(output / f"{split}.bin", mode="r", dtype=np.uint16)
        mask = np.memmap(output / f"{split}.mask.bin", mode="r", dtype=np.uint8)
        assert len(tokens) == len(mask)
        assert set(np.unique(mask)).issubset({0, 1})
        assert int(mask.sum()) > 0

    with pytest.raises(FileExistsError):
        prepare_token_streams([corpus], tokenizer_path, output, valid_fraction=0.01)


def test_assistant_loss_mask_excludes_user_prompt_tokens(tmp_path):
    corpus = tmp_path / "conversations.jsonl"
    conversations = [
        {"messages": [
            {"role": "user", "content": "What is a star?"},
            {"role": "assistant", "content": "A star is a luminous sphere of plasma."},
        ]},
        {"messages": [
            {"role": "user", "content": "Explain gravity."},
            {"role": "assistant", "content": "Gravity is the attraction between masses."},
        ]},
    ]
    corpus.write_text(
        "\n".join(json.dumps(item) for item in conversations),
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "conversation-tokenizer.json"
    train_tokenizer(
        [corpus], tokenizer_path, vocab_size=400, min_frequency=1,
        validation_fraction=0.01,
    )
    output = tmp_path / "assistant-targets"

    report = prepare_token_streams(
        [corpus], tokenizer_path, output,
        valid_fraction=0.01, loss_mask_mode="assistant",
    )
    assert report["loss_mask_mode"] == "assistant"
    assert report["train_target_tokens"] > 0
    assert report["valid_target_tokens"] > 0

    for split in ("train", "valid"):
        tokens = np.memmap(output / f"{split}.bin", mode="r", dtype=np.uint16)
        mask = np.memmap(output / f"{split}.mask.bin", mode="r", dtype=np.uint8)
        assert len(tokens) == len(mask)
        assert 0 < int(mask.sum()) < len(mask)


def test_structured_conversations_use_dedicated_role_and_end_tokens(tmp_path):
    corpus = tmp_path / "control-tokens.jsonl"
    records = [
        {"messages": [
            {"role": "user", "content": "What is a star?"},
            {"role": "assistant", "content": "A star is a luminous sphere of plasma."},
        ]},
        {"messages": [
            {"role": "user", "content": "Explain gravity."},
            {"role": "assistant", "content": "Gravity attracts masses."},
        ]},
    ]
    corpus.write_text(
        "\n".join(json.dumps(item) for item in records),
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "control-tokenizer.json"
    trained = train_tokenizer(
        [corpus], tokenizer_path, vocab_size=400, min_frequency=1,
        validation_fraction=0.01,
    )
    report = prepare_token_streams(
        [corpus], tokenizer_path, tmp_path / "control-streams",
        valid_fraction=0.01, loss_mask_mode="assistant",
    )

    user_id = trained.token_to_id("<|user|>")
    assistant_id = trained.token_to_id("<|assistant|>")
    end_id = trained.token_to_id("<|end|>")
    assert user_id is not None and assistant_id is not None and end_id is not None

    for split in ("train", "valid"):
        token_ids = np.memmap(
            tmp_path / "control-streams" / f"{split}.bin",
            mode="r", dtype=np.uint16,
        )
        mask = np.memmap(
            tmp_path / "control-streams" / f"{split}.mask.bin",
            mode="r", dtype=np.uint8,
        )
        user_positions = np.flatnonzero(token_ids == user_id)
        assistant_positions = np.flatnonzero(token_ids == assistant_id)
        assert user_positions.size == 1
        assert assistant_positions.size == 1
        assert mask[user_positions[0]] == 0
        assert mask[assistant_positions[0]] == 1
        assert mask[np.flatnonzero(token_ids == end_id)[-1]] == 1
    assert report["train_target_tokens"] > 0


def test_chat_ending_with_unanswered_user_does_not_train_document_eos(tmp_path):
    corpus = tmp_path / "tokenizer-source.jsonl"
    rows = [
        {"text": "First independent tokenizer training document with sufficient words."},
        {"text": "Second independent tokenizer training document with sufficient words."},
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    tokenizer = train_tokenizer(
        [corpus], tmp_path / "tokenizer.json", vocab_size=300, min_frequency=1
    )

    conversation = {
        "messages": [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier assistant answer"},
            {"role": "user", "content": "A final unanswered question"},
        ]
    }
    encoded = _encode_record(
        conversation,
        tokenizer,
        special_token_id(tokenizer, "<bos>"),
        special_token_id(tokenizer, "<eos>"),
        "auto",
    )
    assert encoded is not None
    token_ids, loss_mask = encoded
    end_id = special_token_id(tokenizer, "<|end|>")
    end_positions = np.flatnonzero(token_ids == end_id)

    assert len(end_positions) == 3
    assert [int(loss_mask[position]) for position in end_positions] == [0, 1, 0]
    assert loss_mask[-1] == 0


def test_chat_ending_with_assistant_trains_document_eos(tmp_path):
    corpus = tmp_path / "tokenizer-source.jsonl"
    rows = [
        {"text": "First independent tokenizer training document with sufficient words."},
        {"text": "Second independent tokenizer training document with sufficient words."},
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    tokenizer = train_tokenizer(
        [corpus], tmp_path / "tokenizer.json", vocab_size=300, min_frequency=1
    )
    conversation = {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }
    encoded = _encode_record(
        conversation,
        tokenizer,
        special_token_id(tokenizer, "<bos>"),
        special_token_id(tokenizer, "<eos>"),
        "auto",
    )
    assert encoded is not None
    _token_ids, loss_mask = encoded
    assert loss_mask[-1] == 1

def test_empty_records_return_a_complete_segment_contract():
    assert _record_segments("") == ([], False, False, False)
    assert _encode_record("", None, 0, 1, "auto") is None


def test_large_top_level_json_array_is_incrementally_decoded(tmp_path):
    from qmini.data import iter_source_records

    source = tmp_path / "large-array.json"
    rows = [
        {"id": index, "text": f"document-{index}-" + ("x" * 1800)}
        for index in range(90)
    ]
    source.write_text(json.dumps(rows), encoding="utf-8")
    parsed = list(iter_source_records([source]))
    assert len(parsed) == len(rows)
    assert parsed[0]["id"] == 0
    assert parsed[-1]["id"] == 89
    assert parsed[-1]["text"].endswith("x" * 1800)


def test_json_object_reader_and_private_source_manifest_paths(tmp_path):
    source = tmp_path / "private-datasets" / "corpus.json"
    source.parent.mkdir()
    rows = [
        {"text": "A first independent source document with enough words."},
        {"text": "A second independent source document with enough words."},
        {"text": "A third independent source document with enough words."},
    ]
    source.write_text(json.dumps(rows), encoding="utf-8")
    object_source = source.parent / "single-object.json"
    object_source.write_text(json.dumps(rows[0]), encoding="utf-8")
    assert len(list(iter_text_records([object_source]))) == 1

    tokenizer_source = tmp_path / "tokenizer-source.jsonl"
    tokenizer_source.write_text(
        "\n".join(json.dumps(item) for item in rows), encoding="utf-8"
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer(
        [tokenizer_source], tokenizer_path, vocab_size=300, min_frequency=1,
        validation_fraction=0.1,
    )

    default_output = tmp_path / "prepared-default"
    prepare_token_streams(
        [source], tokenizer_path, default_output, valid_fraction=0.1, loss_mask_mode="all"
    )
    manifest_text = (default_output / "dataset.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["source_files"][0]["path"] == source.name
    assert "absolute_path" not in manifest["source_files"][0]
    assert str(tmp_path) not in manifest_text
    assert manifest["source_files"][0]["sha256"]

    explicit_output = tmp_path / "prepared-explicit"
    prepare_token_streams(
        [source], tokenizer_path, explicit_output, valid_fraction=0.1,
        loss_mask_mode="all", include_absolute_source_paths=True,
    )
    explicit = json.loads(
        (explicit_output / "dataset.json").read_text(encoding="utf-8")
    )
    assert explicit["source_files"][0]["absolute_path"] == str(source.resolve())
    assert explicit["absolute_source_paths_included"] is True


def test_simhash_near_duplicates_stay_in_same_train_validation_split(tmp_path):
    from qmini.data import _simhash64

    near_a = "The red fox quickly jumps over the small brown fence near the river."
    near_b = "The red fox quickly jumps over the small brown fence near the river!"
    assert near_a != near_b
    assert _simhash64(near_a) == _simhash64(near_b)

    rows = [
        {"text": near_a},
        {"text": near_b},
        {"text": "Astronomers detected a distant galaxy with a powerful radio telescope."},
        {"text": "Marine biologists studied coral reefs in the warm tropical ocean."},
        {"text": "Engineers designed a lightweight bridge using strong composite materials."},
        {"text": "Botanists measured how desert plants conserve water during long dry seasons."},
    ]
    corpus = tmp_path / "near-duplicates.jsonl"
    corpus.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "near-duplicate-tokenizer.json"
    tokenizer = train_tokenizer(
        [corpus],
        tokenizer_path,
        vocab_size=320,
        min_frequency=1,
        validation_fraction=0.25,
        near_duplicate_distance=3,
    )

    output = tmp_path / "near-duplicate-streams"
    report = prepare_token_streams(
        [corpus],
        tokenizer_path,
        output,
        valid_fraction=0.25,
        loss_mask_mode="all",
        near_duplicate_distance=3,
    )
    assert report["unique_documents"] == 6
    assert report["near_duplicate_groups"] == 5
    assert report["near_duplicate_documents_grouped"] == 1
    assert report["near_duplicate_hamming_distance"] == 3
    assert report["deduplication"] == "exact-document-sha256+simhash64"

    split_texts = {}
    for split in ("train", "valid"):
        token_ids = np.fromfile(output / f"{split}.bin", dtype=np.uint16)
        split_texts[split] = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)

    appearances = {
        split: ("red fox quickly jumps over the small brown fence" in text)
        for split, text in split_texts.items()
    }
    assert sum(appearances.values()) == 1, "Near-duplicate documents leaked across splits"


def test_near_duplicate_distance_validation_is_strict():
    from qmini.data import _validate_near_duplicate_distance

    for distance in (0, 1, 2, 3):
        _validate_near_duplicate_distance(distance)
    for distance in (-1, 4, 1.5, True):
        with pytest.raises(ValueError, match="0 to 3"):
            _validate_near_duplicate_distance(distance)


def test_prepare_rejects_tokenizer_and_dataset_split_mismatch(tmp_path):
    corpus = tmp_path / "split-contract.jsonl"
    rows = [
        {"text": "An independent document about galaxies and stellar evolution."},
        {"text": "A separate document about oceans and marine ecosystems."},
        {"text": "A different document about materials and bridge construction."},
    ]
    corpus.write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    tokenizer_path = tmp_path / "split-contract-tokenizer.json"
    train_tokenizer(
        [corpus], tokenizer_path, vocab_size=320, min_frequency=1,
        validation_fraction=0.1, near_duplicate_distance=2,
    )

    with pytest.raises(ValueError, match="near-duplicate thresholds"):
        prepare_token_streams(
            [corpus], tokenizer_path, tmp_path / "mismatched-output",
            valid_fraction=0.1, loss_mask_mode="all", near_duplicate_distance=1,
        )


def test_large_json_array_trailing_whitespace_is_consumed_in_bounded_chunks(tmp_path):
    import io
    from qmini.data import _iter_json_array

    class TrackingReader(io.StringIO):
        def __init__(self, value):
            super().__init__(value)
            self.read_sizes = []

        def read(self, size=-1):
            self.read_sizes.append(size)
            return super().read(size)

    reader = TrackingReader('[{"value": 1}]' + (" " * 100_000))
    parsed = list(_iter_json_array(reader, tmp_path / "large.json", chunk_chars=256))
    assert parsed == [{"value": 1}]
    assert reader.read_sizes
    assert all(0 < size <= 256 for size in reader.read_sizes)

    invalid = TrackingReader('[{"value": 1}]' + (" " * 200) + "unexpected")
    with pytest.raises(ValueError, match="Unexpected data after JSON array"):
        list(_iter_json_array(invalid, tmp_path / "invalid.json", chunk_chars=64))
    assert all(0 < size <= 64 for size in invalid.read_sizes)


@pytest.mark.parametrize(
    "prefix",
    [
        " ",
        "\n\n",
        " \t\r\n  ",
    ],
)
def test_streaming_json_array_accepts_leading_whitespace(tmp_path, prefix):
    source = tmp_path / "leading-whitespace.json"
    source.write_text(
        prefix + '[{"text":"first record"},{"text":"second record"}]',
        encoding="utf-8",
    )
    records = list(iter_text_records([source]))
    assert records == ["first record", "second record"]


def test_streaming_json_array_accepts_utf8_bom_and_leading_whitespace(tmp_path):
    source = tmp_path / "bom-leading-whitespace.json"
    source.write_bytes(
        b"\xef\xbb\xbf \r\n\t ["
        b'{"text":"first after BOM"},{"text":"second after BOM"}]'
    )
    assert list(iter_text_records([source])) == [
        "first after BOM", "second after BOM"
    ]


def test_structured_message_bodies_escape_reserved_token_spellings(tmp_path):
    from qmini.text import SPECIAL_TOKENS, escape_special_token_literals

    literal_text = "Quote " + " ".join(SPECIAL_TOKENS) + " as ordinary text."
    record = {"messages": [
        {"role": "user", "content": literal_text},
        {"role": "assistant", "content": "The markers above are quoted text."},
    ]}
    segments, structured, has_assistant, final_is_assistant = _record_segments(record)
    assert structured and has_assistant and final_is_assistant
    body_segments = [segment for segment, _ in segments if segment not in {
        "<|user|>", "<|assistant|>", "<|end|>\n"
    }]
    combined = "".join(body_segments)
    assert all(token not in combined for token in SPECIAL_TOKENS)
    assert "\u200b" in combined
    assert escape_special_token_literals(literal_text) in combined


def test_escaped_reserved_text_does_not_encode_as_control_token_ids(tmp_path):
    from qmini.text import SPECIAL_TOKENS, escape_special_token_literals

    corpus = tmp_path / "literal-specials.jsonl"
    literal = "This document quotes " + " ".join(SPECIAL_TOKENS) + " literally."
    records = [
        {"messages": [
            {"role": "user", "content": literal},
            {"role": "assistant", "content": "They are examples, not instructions."},
        ]},
        {"messages": [
            {"role": "user", "content": "A different question."},
            {"role": "assistant", "content": "A different answer."},
        ]},
    ]
    corpus.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    tokenizer_path = tmp_path / "literal-specials-tokenizer.json"
    tokenizer = train_tokenizer(
        [corpus], tokenizer_path, vocab_size=300, min_frequency=1,
        validation_fraction=0.25,
    )
    escaped = escape_special_token_literals(literal)
    encoded_ids = set(tokenizer.encode(escaped).ids)
    reserved_ids = {
        tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
        if tokenizer.token_to_id(token) is not None
    }
    assert encoded_ids.isdisjoint(reserved_ids)


def test_release_source_provenance_is_deterministic_and_rejects_unknown_license(tmp_path):
    import json
    from qmini.data import build_source_provenance

    corpus = tmp_path / "licensed-corpus.txt"
    corpus.write_text("First distinct document.\n\nSecond distinct document.\n", encoding="utf-8")
    metadata = tmp_path / "sources.json"
    payload = {
        "format": "qmini-source-provenance-v1",
        "sources": [{
            "path": corpus.name,
            "source_name": "Example public-domain research corpus",
            "source_uri": "https://example.org/corpus",
            "license_spdx": "CC-BY-4.0",
            "collected_at": "2026-11-10",
            "language": "en",
            "transformation_version": "raw-v1",
        }],
    }
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    files_a, digest_a, screened_a = build_source_provenance(
        [corpus], metadata_path=metadata, release_mode=True
    )
    files_b, digest_b, screened_b = build_source_provenance(
        [corpus], metadata_path=metadata, release_mode=True
    )
    assert digest_a == digest_b
    assert files_a == files_b
    assert screened_a == screened_b == 0
    assert files_a[0]["license_spdx"] == "CC-BY-4.0"
    assert files_a[0]["sha256"]
    assert files_a[0]["size_bytes"] == corpus.stat().st_size

    payload["sources"][0]["license_spdx"] = "UNKNOWN"
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown or unapproved license"):
        build_source_provenance([corpus], metadata_path=metadata, release_mode=True)


def test_release_source_provenance_blocks_pii_without_logging_raw_content(tmp_path):
    import json
    from qmini.data import build_source_provenance

    corpus = tmp_path / "pii-corpus.txt"
    private_value = "private.person@example.org"
    corpus.write_text(
        "A document containing " + private_value + " must not enter a release corpus.\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "sources.json"
    metadata.write_text(json.dumps({
        "format": "qmini-source-provenance-v1",
        "sources": [{
            "path": corpus.name,
            "source_name": "Test corpus",
            "source_uri": "https://example.org/corpus",
            "license_spdx": "MIT",
            "collected_at": "2026-11-10",
            "language": "en",
            "transformation_version": "raw-v1",
        }],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="potential PII/secrets") as error:
        build_source_provenance([corpus], metadata_path=metadata, release_mode=True)
    assert private_value not in str(error.value)


def test_release_provenance_is_bound_to_tokenizer_and_prepared_dataset(tmp_path):
    import json
    from qmini.data import prepare_token_streams
    from qmini.tokenizer import train_tokenizer

    corpus = tmp_path / "release-corpus.txt"
    corpus.write_text(
        "Q-Mini trains only from random initialization.\n\n"
        "Every data source needs a recorded license and transformation.\n\n"
        "Validation documents stay separate from tokenizer fitting.\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "sources.json"
    metadata.write_text(json.dumps({
        "format": "qmini-source-provenance-v1",
        "sources": [{
            "path": corpus.name,
            "source_name": "Q-Mini release-gate fixture",
            "source_uri": "https://example.org/qmini-fixture",
            "license_spdx": "CC0-1.0",
            "collected_at": "2026-11-10",
            "language": "en",
            "transformation_version": "fixture-v1",
        }],
    }), encoding="utf-8")

    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer(
        [corpus],
        tokenizer_path,
        vocab_size=300,
        min_frequency=1,
        validation_fraction=0.25,
        source_metadata_path=metadata,
        release_mode=True,
    )
    tokenizer_manifest = json.loads(
        (tmp_path / "tokenizer.json.qmini.json").read_text(encoding="utf-8")
    )
    assert tokenizer_manifest["release_mode"] is True
    assert tokenizer_manifest["source_provenance_sha256"]

    report = prepare_token_streams(
        [corpus],
        tokenizer_path,
        tmp_path / "prepared",
        valid_fraction=0.25,
        source_metadata_path=metadata,
        release_mode=True,
    )
    assert report["release_mode"] is True
    assert report["source_provenance_sha256"] == tokenizer_manifest["source_provenance_sha256"]
    assert report["source_files"][0]["document_count"] == 3
    assert report["source_files"][0]["license_spdx"] == "CC0-1.0"

    from qmini.train import _verify_prepared_dataset_manifest
    prepared_dir = tmp_path / "prepared"
    assert _verify_prepared_dataset_manifest(
        prepared_dir / "train.bin",
        prepared_dir / "valid.bin",
        tokenizer_path,
        prepared_dir / "train.mask.bin",
        prepared_dir / "valid.mask.bin",
        tokenizer.get_vocab_size(),
    ) == prepared_dir / "dataset.json"

    manifest_path = prepared_dir / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_files"][0]["license_spdx"] = "MIT"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source provenance hash is invalid"):
        _verify_prepared_dataset_manifest(
            prepared_dir / "train.bin",
            prepared_dir / "valid.bin",
            tokenizer_path,
            prepared_dir / "train.mask.bin",
            prepared_dir / "valid.mask.bin",
            tokenizer.get_vocab_size(),
        )

    Path(str(tokenizer_path) + ".qmini.json").unlink()
    with pytest.raises(ValueError, match="tokenizer provenance sidecar"):
        prepare_token_streams(
            [corpus],
            tokenizer_path,
            tmp_path / "prepared-without-sidecar",
            valid_fraction=0.25,
            source_metadata_path=metadata,
            release_mode=True,
        )
