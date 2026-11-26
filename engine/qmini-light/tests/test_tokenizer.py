import hashlib

import pytest

from qmini.data import iter_training_text_records
from qmini.tokenizer import load_tokenizer, train_tokenizer


def _is_validation(text: str, fraction: float) -> bool:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / (2**64) < fraction


def test_local_byte_bpe_round_trip_and_no_overwrite(tmp_path):
    corpus = tmp_path / "corpus.txt"
    text = (
        "Q-Mini learns carefully. Namaste नमस्ते 🌌\n"
        "Small models need clean data."
    )
    corpus.write_text(
        text + "\n\nA second independent training document for the tokenizer.",
        encoding="utf-8",
    )
    output = tmp_path / "tokenizer.json"

    trained = train_tokenizer(
        [corpus], output, vocab_size=300, min_frequency=1,
        validation_fraction=0.02,
    )
    loaded = load_tokenizer(output)
    encoded = loaded.encode(text)
    assert encoded.ids
    assert loaded.decode(encoded.ids) == text
    assert trained.get_vocab_size() <= 300
    for token in ("<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|other|>", "<|end|>"):
        assert loaded.token_to_id(token) is not None

    with pytest.raises(FileExistsError):
        train_tokenizer([corpus], output, vocab_size=300, min_frequency=1)


def test_tokenizer_iterator_excludes_validation_and_duplicate_documents(tmp_path):
    train_doc = next(
        f"training source document {index} contains enough words and details"
        for index in range(100)
        if not _is_validation(
            f"training source document {index} contains enough words and details",
            0.49,
        )
    )
    valid_doc = next(
        f"held out validation document {index} contains unique held out material"
        for index in range(100)
        if _is_validation(
            f"held out validation document {index} contains unique held out material",
            0.49,
        )
    )
    corpus = tmp_path / "mixed.txt"
    corpus.write_text(
        train_doc + "\n\n" + valid_doc + "\n\n" + train_doc,
        encoding="utf-8",
    )
    rows = list(iter_training_text_records([corpus], valid_fraction=0.49))
    assert rows == [train_doc]
    assert valid_doc not in rows


def test_training_record_iterator_can_resume_on_another_thread(tmp_path):
    # tokenizers.train_from_iterator may resume a Python generator from a
    # worker thread. Reproduce that thread hop directly while the generator
    # keeps one SQLite connection open across yields.
    from concurrent.futures import ThreadPoolExecutor

    corpus = tmp_path / "thread-hop.txt"
    corpus.write_text(
        "\n\n".join(
            f"independent training document number {index} contains distinct words and details"
            for index in range(40)
        ),
        encoding="utf-8",
    )

    iterator = iter_training_text_records([corpus], valid_fraction=0.02)
    first = next(iterator)  # Opens and builds the SQLite index in this thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        second = executor.submit(next, iterator).result()  # Resume in another thread.

    remaining = list(iterator)  # Resume again in the original thread and close.
    rows = [first, second, *remaining]
    assert len(rows) >= 2
    assert len(rows) == len(set(rows))
    assert all(row.strip() for row in rows)


def test_tokenizer_matches_rebalanced_prepare_split(tmp_path):
    from qmini.data import prepare_token_streams

    train_docs = []
    for index in range(100):
        text = f"training document {index} contains unique words for tokenizer split validation"
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        score = int.from_bytes(digest[:8], "big") / (2**64)
        if score >= 0.02:
            train_docs.append((digest, text))
        if len(train_docs) == 2:
            break
    assert len(train_docs) == 2
    # Both docs initially hash to train. Preparation moves the smallest digest to validation.
    expected_train = max(train_docs, key=lambda pair: pair[0])[1]
    source = tmp_path / "small-corpus.txt"
    source.write_text(train_docs[0][1] + "\n\n" + train_docs[1][1], encoding="utf-8")

    tokenizer_rows = list(iter_training_text_records([source], valid_fraction=0.02))
    assert tokenizer_rows == [expected_train]

    from qmini.tokenizer import train_tokenizer
    tokenizer_path = tmp_path / "small-tokenizer.json"
    train_tokenizer([source], tokenizer_path, vocab_size=300, min_frequency=1)
    report = prepare_token_streams(
        [source], tokenizer_path, tmp_path / "small-token-streams",
        valid_fraction=0.02, loss_mask_mode="all",
    )
    assert report["train_documents"] == 1
    assert report["valid_documents"] == 1

def test_tokenizer_refuses_empty_training_partition(tmp_path):
    validation_only = next(
        f"held out only document {index} must not be used for tokenizer training"
        for index in range(100)
        if _is_validation(
            f"held out only document {index} must not be used for tokenizer training",
            0.49,
        )
    )
    corpus = tmp_path / "validation.txt"
    corpus.write_text(validation_only, encoding="utf-8")
    with pytest.raises(ValueError, match="No unique training documents"):
        train_tokenizer(
            [corpus], tmp_path / "tokenizer.json",
            vocab_size=300, min_frequency=1, validation_fraction=0.49,
        )


def test_zero_validation_fraction_is_rejected_at_tokenizer_api_boundary(tmp_path):
    corpus = tmp_path / "zero-holdout.txt"
    corpus.write_text(
        "A first independent document for tokenizer fitting.\n\n"
        "A second independent document for tokenizer fitting.",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="greater than 0 and below 0.5"):
        train_tokenizer(
            [corpus],
            tmp_path / "tokenizer.json",
            vocab_size=300,
            min_frequency=1,
            validation_fraction=0.0,
        )
