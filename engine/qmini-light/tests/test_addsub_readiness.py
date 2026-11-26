import pytest

from qmini_addsub.evaluate import _answer_diversity
from qmini_addsub.readiness import assess_addsub_readiness


def _assess(metrics, heldout, checkpoint_sha256="f" * 64):
    return assess_addsub_readiness(
        metrics, heldout, checkpoint_sha256=checkpoint_sha256
    )


def _metrics():
    identity = {
        "train_sha256": "a" * 64,
        "valid_sha256": "b" * 64,
        "test_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "data_manifest_sha256": "e" * 64,
    }
    return [
        {"step": 50, "train_loss": 4.0, "valid_response_loss": 3.0,
         "learning_rate": 1e-5, **identity},
        {"step": 100, "train_loss": 3.5, "valid_response_loss": 2.9,
         "learning_rate": 1e-6, **identity},
    ]


def _heldout():
    return {
        "format": "qmini-addsub-heldout-eval-v1",
        "checkpoint_sha256": "f" * 64,
        "tokenizer_sha256": "d" * 64,
        "dataset_manifest_sha256": "e" * 64,
        "test_sha256": "c" * 64,
        "test_records": 50,
        "generation_failures": 0,
        "mean_token_f1": 0.30,
        "mean_prediction_tokens": 9.0,
        "output_diversity": _answer_diversity(
            [f"Answer number {index}" for index in range(50)]
        ),
    }


def test_addsub_readiness_requires_validation_learning_and_noncollapsed_heldout_answers():
    result = _assess(_metrics(), _heldout())
    assert result["ready"] is True
    assert result["status"] == "ready_for_review"
    assert result["checks_passed"] == result["checks_total"] == 3


def test_addsub_answer_diversity_detects_same_answer_with_punctuation_variants():
    result = _answer_diversity([
        "Same answer!",
        "same answer.",
        "A different answer",
        "",
    ])
    assert result["examples"] == 4
    assert result["unique_normalized_answers"] == 3
    assert result["unique_answer_fraction"] == 0.75
    assert result["largest_identical_answer_count"] == 2
    assert result["largest_identical_answer_fraction"] == 0.5
    assert result["empty_answer_count"] == 1
    assert result["all_outputs_identical"] is False


def test_addsub_readiness_fails_for_constant_outputs_even_when_lexical_score_is_high():
    heldout = _heldout()
    heldout["output_diversity"] = _answer_diversity(["same answer"] * 50)
    result = _assess(_metrics(), heldout)
    check = next(item for item in result["checks"] if item["name"] == "heldout_repetition_collapse")
    assert check["passed"] is False
    assert result["ready"] is False



def test_addsub_readiness_fails_for_phrase_loops_inside_unique_answers():
    repeated = "this phrase must not loop forever "
    answers = [repeated * 5 + f"unique ending {index}" for index in range(50)]
    heldout = _heldout()
    heldout["output_diversity"] = _answer_diversity(answers)
    result = _assess(_metrics(), heldout)
    check = next(item for item in result["checks"] if item["name"] == "heldout_repetition_collapse")
    assert check["passed"] is False
    assert check["observed"]["max_repeated_4gram_count"] > 4
    assert result["ready"] is False

def test_addsub_readiness_fails_when_validation_response_loss_does_not_improve():
    metrics = _metrics()
    metrics[-1]["valid_response_loss"] = 3.1
    result = _assess(metrics, _heldout())
    check = next(item for item in result["checks"] if item["name"] == "sft_validation_learning")
    assert check["passed"] is False
    assert result["ready"] is False


def test_addsub_readiness_fails_on_generation_failure_and_insufficient_test_set():
    heldout = _heldout()
    heldout["generation_failures"] = 1
    heldout["test_records"] = 10
    heldout["output_diversity"] = _answer_diversity([f"Answer {i}" for i in range(10)])
    result = _assess(_metrics(), heldout)
    check = next(item for item in result["checks"] if item["name"] == "heldout_response_quality")
    assert check["passed"] is False
    assert result["ready"] is False


def test_addsub_readiness_rejects_heldout_test_split_hash_mismatch():
    heldout = _heldout()
    heldout["test_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="test_sha256 does not match"):
        _assess(_metrics(), heldout)


def test_addsub_readiness_requires_two_validation_evaluations():
    with pytest.raises(ValueError, match="metrics"):
        _assess([], _heldout())
    result = _assess(_metrics()[:1], _heldout())
    assert result["ready"] is False


def test_addsub_readiness_rejects_a_report_from_a_different_checkpoint():
    with pytest.raises(ValueError, match="checkpoint_sha256 does not match"):
        _assess(_metrics(), _heldout(), checkpoint_sha256="0" * 64)
