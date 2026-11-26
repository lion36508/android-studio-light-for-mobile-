from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from qmini_addsub.conditioning import (
    build_conditioning_report,
    total_variation_distance,
)


class _Encoded:
    def __init__(self, text: str):
        spans = list(re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE))
        self.offsets = [(match.start(), match.end()) for match in spans]
        self.ids = [
            8 + (sum(ord(char) for char in match.group(0)) % 500)
            for match in spans
        ]


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return _Encoded(text)

    def token_to_id(self, token: str):
        return {
            "<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3,
            "<|system|>": 4, "<|user|>": 5,
            "<|assistant|>": 6, "<|end|>": 7,
        }.get(token)

    def id_to_token(self, token_id: int):
        return f"token-{token_id}"


class _DeterministicProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_context=512, vocab_size=512)

    def init_cache(self, batch_size=1, device=None):
        return torch.zeros((), dtype=torch.long, device=device)

    def forward_step(self, input_ids, caches):
        updated = caches + input_ids.to(dtype=torch.long).sum()
        phase = int(updated.item()) % 13
        positions = torch.arange(self.config.vocab_size, device=input_ids.device)
        logits = torch.sin((positions.float() + phase) * 0.071).unsqueeze(0)
        return logits, updated


def test_total_variation_distance_is_symmetric_and_bounded():
    left = [0.7, 0.2, 0.1]
    right = [0.1, 0.2, 0.7]
    assert total_variation_distance(left, right) == pytest.approx(0.6)
    assert total_variation_distance(left, right) == pytest.approx(
        total_variation_distance(right, left)
    )
    assert 0.0 <= total_variation_distance(left, right) <= 1.0


@pytest.mark.parametrize(
    "left,right",
    [
        ([1.0, float("nan")], [0.5, 0.5]),
        ([1.0, -0.1], [0.5, 0.5]),
        ([0.0, 0.0], [0.5, 0.5]),
        ([1.0], [0.5, 0.5]),
        ([[0.5, 0.5]], [[0.5, 0.5]]),
    ],
)
def test_total_variation_distance_rejects_malformed_inputs(left, right):
    with pytest.raises(ValueError):
        total_variation_distance(left, right)


def test_conditioning_report_measures_question_and_context_sensitivity():
    report = build_conditioning_report(
        _DeterministicProbeModel(),
        _Tokenizer(),
        question_prompts=["Hi.", "What is 2 + 2?", "Zorp blip fardle?"],
        context_distances=[0, 3],
        top_tokens=4,
    )
    assert report["format"] == "qmini-addsub-conditioning-probe-v1"
    assert report["question_prompt_count"] == 3
    assert report["question_pairwise_tvd_summary"]["pair_count"] == 3
    assert len(report["question_pairwise_tvd"]) == 3
    assert report["question_pairwise_tvd_summary"]["maximum"] > 0.0
    assert len(report["context_distance_probe"]) == 2
    assert all(row["tvd"] > 0.0 for row in report["context_distance_probe"])
    for row in report["context_distance_probe"]:
        assert row["left_distance"]["tokens_after_changed_word_to_assistant_boundary"] >= 2
        assert row["right_distance"]["tokens_after_changed_word_to_assistant_boundary"] >= 2
        assert 0.0 <= row["tvd"] <= 1.0
    for row in report["question_prompts"]:
        assert len(row["top_tokens"]) == 4
        assert row["entropy_nats"] > 0.0
