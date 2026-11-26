from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from tokenizers import Tokenizer

from .data import escape_special_token_literals
from .generate import build_chat_prompt, load_addsub_model
from .sft import SYSTEM_TEXT
from .train import sha256_file


DEFAULT_QUESTION_PROMPTS = (
    "Hi.",
    "Count from 1 to 10.",
    "What is your favourite colour?",
    "What is 2 + 2?",
    "Zorp blip fardle?",
)
DEFAULT_CONTEXT_DISTANCES = (0, 1, 3, 5, 10, 20, 40)
_CONTEXT_PREFIX = "Neutral context record. The target value is: "
_CONTEXT_LEFT = "cat"
_CONTEXT_RIGHT = "dog"
_CONTEXT_FILLER = " quiet"


def total_variation_distance(
    left: Sequence[float] | torch.Tensor,
    right: Sequence[float] | torch.Tensor,
) -> float:
    """Return TVD for two finite, non-negative one-dimensional distributions."""
    p = torch.as_tensor(left, dtype=torch.float64, device="cpu")
    q = torch.as_tensor(right, dtype=torch.float64, device="cpu")
    if p.ndim != 1 or q.ndim != 1 or p.numel() < 1 or p.shape != q.shape:
        raise ValueError("TVD inputs must be non-empty one-dimensional vectors with equal shape.")
    if not bool(torch.isfinite(p).all()) or not bool(torch.isfinite(q).all()):
        raise ValueError("TVD inputs must contain only finite values.")
    if bool((p < 0).any()) or bool((q < 0).any()):
        raise ValueError("TVD inputs must be non-negative.")
    p_sum = float(p.sum().item())
    q_sum = float(q.sum().item())
    if p_sum <= 0.0 or q_sum <= 0.0:
        raise ValueError("TVD inputs must each have positive total mass.")
    p = p / p_sum
    q = q / q_sum
    return float((0.5 * torch.abs(p - q).sum()).item())


def _entropy(probabilities: torch.Tensor) -> float:
    values = probabilities.to(dtype=torch.float64, device="cpu")
    positive = values > 0
    return float((-(values[positive] * torch.log(values[positive])).sum()).item())


def _context_pair(distance_words: int) -> tuple[str, str, tuple[int, int]]:
    """Create equal-template prompts differing at one controlled word position."""
    if isinstance(distance_words, bool) or not isinstance(distance_words, int) or distance_words < 0:
        raise ValueError("Context word distance must be a non-negative integer.")
    left = _CONTEXT_PREFIX + _CONTEXT_LEFT + (_CONTEXT_FILLER * distance_words)
    right = _CONTEXT_PREFIX + _CONTEXT_RIGHT + (_CONTEXT_FILLER * distance_words)
    span = (len(_CONTEXT_PREFIX), len(_CONTEXT_PREFIX) + len(_CONTEXT_LEFT))
    return left, right, span


def _tokens_after_changed_span(
    tokenizer: Tokenizer,
    text: str,
    changed_span: tuple[int, int],
) -> dict[str, int]:
    start, end = changed_span
    if not 0 <= start < end <= len(text):
        raise ValueError("Changed character span is outside the prompt.")
    encoded = tokenizer.encode(escape_special_token_literals(text), add_special_tokens=False)
    affected = [
        index
        for index, (token_start, token_end) in enumerate(encoded.offsets)
        if token_start < end and token_end > start
    ]
    if not affected:
        raise ValueError("Tokenizer produced no token overlapping the changed word.")
    # The generated assistant boundary follows the user message's end token
    # and assistant role token, so include both control tokens in this count.
    after = len(encoded.ids) - max(affected) - 1 + 2
    return {
        "changed_word_token_count": len(affected),
        "tokens_after_changed_word_to_assistant_boundary": after,
        "user_content_tokens": len(encoded.ids),
    }


@torch.inference_mode()
def _next_token_probabilities(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    user_prompt: str,
    system_text: str = SYSTEM_TEXT,
) -> tuple[torch.Tensor, int]:
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("Each conditioning prompt must be a non-empty string.")
    model.eval()
    device = next(model.parameters()).device
    prompt_ids = build_chat_prompt(
        tokenizer,
        [{"role": "user", "content": user_prompt}],
        model.config.max_context,
        system_text=system_text,
    )
    if not prompt_ids:
        raise RuntimeError("Conditioning probe encoded an empty model prompt.")
    caches = model.init_cache(batch_size=1, device=device)
    logits = None
    for token_id in prompt_ids:
        input_ids = torch.tensor([token_id], dtype=torch.long, device=device)
        logits, caches = model.forward_step(input_ids, caches)
    if not isinstance(logits, torch.Tensor) or logits.shape != (1, model.config.vocab_size):
        raise RuntimeError("Model returned an invalid next-token logits shape.")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Model returned non-finite next-token logits.")
    probabilities = torch.softmax(logits[0].float(), dim=-1).to(device="cpu")
    if not bool(torch.isfinite(probabilities).all()) or not torch.isclose(
        probabilities.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5
    ):
        raise RuntimeError("Next-token probabilities failed normalization checks.")
    return probabilities, len(prompt_ids)


def _top_tokens(
    probabilities: torch.Tensor, tokenizer: Tokenizer, count: int
) -> list[dict[str, Any]]:
    count = min(count, probabilities.numel())
    values, indices = torch.topk(probabilities, count)
    return [
        {
            "token_id": int(index),
            "token": tokenizer.id_to_token(int(index)),
            "probability": float(value),
        }
        for value, index in zip(values.tolist(), indices.tolist())
    ]


@torch.inference_mode()
def build_conditioning_report(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    question_prompts: Sequence[str] = DEFAULT_QUESTION_PROMPTS,
    context_distances: Sequence[int] = DEFAULT_CONTEXT_DISTANCES,
    *,
    system_text: str = SYSTEM_TEXT,
    top_tokens: int = 10,
) -> dict[str, Any]:
    """Measure prompt-boundary distribution sensitivity; this is not a quality score."""
    prompts = list(question_prompts)
    if len(prompts) < 2 or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Provide at least two non-empty question prompts.")
    if len(set(prompts)) != len(prompts):
        raise ValueError("Question prompts must be unique so pairwise comparisons are meaningful.")
    if isinstance(top_tokens, bool) or not isinstance(top_tokens, int) or top_tokens < 1:
        raise ValueError("top_tokens must be a positive integer.")
    distances = list(context_distances)
    if not distances or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in distances
    ):
        raise ValueError("context_distances must contain non-negative integers.")
    if len(set(distances)) != len(distances):
        raise ValueError("context_distances must not contain duplicates.")

    distributions: list[torch.Tensor] = []
    question_rows: list[dict[str, Any]] = []
    for prompt in prompts:
        probabilities, prompt_tokens = _next_token_probabilities(
            model, tokenizer, prompt, system_text
        )
        distributions.append(probabilities)
        question_rows.append({
            "prompt": prompt,
            "prompt_tokens_including_control_tokens": prompt_tokens,
            "entropy_nats": _entropy(probabilities),
            "top_tokens": _top_tokens(probabilities, tokenizer, top_tokens),
        })

    pairwise_rows = []
    pairwise_values = []
    for left_index in range(len(prompts)):
        for right_index in range(left_index + 1, len(prompts)):
            distance = total_variation_distance(
                distributions[left_index], distributions[right_index]
            )
            pairwise_values.append(distance)
            pairwise_rows.append({
                "left_prompt_index": left_index,
                "right_prompt_index": right_index,
                "left_prompt": prompts[left_index],
                "right_prompt": prompts[right_index],
                "tvd": distance,
            })

    context_rows = []
    for distance_words in distances:
        left_prompt, right_prompt, changed_span = _context_pair(distance_words)
        left_probabilities, left_prompt_tokens = _next_token_probabilities(
            model, tokenizer, left_prompt, system_text
        )
        right_probabilities, right_prompt_tokens = _next_token_probabilities(
            model, tokenizer, right_prompt, system_text
        )
        context_rows.append({
            "requested_filler_words_after_change": distance_words,
            "left_prompt": left_prompt,
            "right_prompt": right_prompt,
            "left_distance": _tokens_after_changed_span(tokenizer, left_prompt, changed_span),
            "right_distance": _tokens_after_changed_span(tokenizer, right_prompt, changed_span),
            "left_full_prompt_tokens": left_prompt_tokens,
            "right_full_prompt_tokens": right_prompt_tokens,
            "tvd": total_variation_distance(left_probabilities, right_probabilities),
        })

    return {
        "format": "qmini-addsub-conditioning-probe-v1",
        "vocabulary_size": int(model.config.vocab_size),
        "system_text": system_text,
        "question_prompt_count": len(prompts),
        "question_prompts": question_rows,
        "question_pairwise_tvd": pairwise_rows,
        "question_pairwise_tvd_summary": {
            "pair_count": len(pairwise_values),
            "minimum": min(pairwise_values),
            "mean": sum(pairwise_values) / len(pairwise_values),
            "maximum": max(pairwise_values),
        },
        "context_distance_probe": context_rows,
        "interpretation_limit": (
            "TVD, entropy and top-token changes measure next-token distribution "
            "sensitivity only. They do not establish semantic understanding, "
            "answer correctness or general language quality. Context probes use "
            "controlled synthetic templates and must be supplemented with "
            "held-out, human-reviewed examples."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Q-Mini Add/Sub next-token sensitivity to prompts and context distance."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Question prompt to probe; repeat at least twice. Uses built-in controls when omitted.",
    )
    parser.add_argument("--system-text", default=SYSTEM_TEXT)
    parser.add_argument("--top-tokens", type=int, default=10)
    parser.add_argument(
        "--context-distances",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_DISTANCES),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    if not checkpoint.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("Checkpoint and tokenizer files are both required.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = load_addsub_model(
        checkpoint,
        device_name=args.device,
        tokenizer_path=tokenizer_path,
    )
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match the checkpoint configuration.")

    report = build_conditioning_report(
        model,
        tokenizer,
        question_prompts=args.prompts or DEFAULT_QUESTION_PROMPTS,
        context_distances=args.context_distances,
        system_text=args.system_text,
        top_tokens=args.top_tokens,
    )
    report["checkpoint_sha256"] = sha256_file(checkpoint)
    report["tokenizer_sha256"] = sha256_file(tokenizer_path)
    report["device"] = args.device

    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError("Conditioning report output already exists; choose a new path.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    print(serialized, end="")


if __name__ == "__main__":
    main()
