from __future__ import annotations

import hashlib
import json

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from qmini_addsub.tokenizer_audit import audit_tokenizer, audit_tokenizer_file


SPECIAL = [
    "<unk>", "<pad>", "<bos>", "<eos>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
]


def _trained_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=320,
        min_frequency=1,
        special_tokens=SPECIAL,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        [
            " Hello, how can I help you?",
            " I'm sorry. Can you please explain?",
            " What is your name?",
            " Count from one to ten.",
        ] * 8,
        trainer=trainer,
    )
    return tokenizer


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bytelevel_base_alphabet_is_not_mislabeled_as_wasted_tokens():
    report = audit_tokenizer(_trained_tokenizer())

    byte_info = report["byte_level_alphabet"]
    assert byte_info["expected_byte_symbols"] == 256
    assert byte_info["present_byte_symbols"] == 256
    assert byte_info["non_ascii_display_glyphs_from_byte_alphabet"] > 0
    assert "not, by themselves, wasted" in byte_info["interpretation"]
    assert report["special_tokens"]["ids_are_unique"] is True


def test_common_chat_word_and_system_prompt_coverage_are_reported():
    report = audit_tokenizer(_trained_tokenizer())

    coverage = report["common_chat_terms"]
    assert coverage["term_count"] >= 20
    assert 0.0 <= coverage["single_token_fraction"] <= 1.0
    assert any(item["term"] == "hello" for item in coverage["probes"])
    assert report["system_prompt_probe"]["token_count"] > 0


def test_validation_probe_requires_matching_manifest_hashes(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    validation_path = tmp_path / "pretrain_validation.txt"
    manifest_path = tmp_path / "dataset_manifest.json"
    _trained_tokenizer().save(str(tokenizer_path))
    validation_path.write_text("Hello, how can I help you?\n", encoding="utf-8")
    manifest = {
        "format": "qmini-addsub-data-v1",
        "pretraining": {
            "splits": {
                "validation": {
                    "path": validation_path.name,
                    "sha256": _sha256(validation_path),
                }
            }
        },
        "tokenizer": {"sha256": _sha256(tokenizer_path)},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_tokenizer_file(
        tokenizer_path,
        validation_text_path=validation_path,
        dataset_manifest_path=manifest_path,
        validation_byte_budget=1024,
    )
    assert report["heldout_validation_probe"]["token_count"] > 0
    assert report["validation_text_sha256"] == _sha256(validation_path)
    assert report["dataset_manifest_sha256"] == _sha256(manifest_path)

    manifest["pretraining"]["splits"]["validation"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the dataset manifest"):
        audit_tokenizer_file(
            tokenizer_path,
            validation_text_path=validation_path,
            dataset_manifest_path=manifest_path,
        )


def test_validation_text_and_manifest_must_be_supplied_together(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    _trained_tokenizer().save(str(tokenizer_path))
    with pytest.raises(ValueError, match="must be supplied together"):
        audit_tokenizer_file(tokenizer_path, validation_text_path=tmp_path / "missing.txt")
