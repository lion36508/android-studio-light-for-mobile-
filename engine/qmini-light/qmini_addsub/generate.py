from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .config import AddSubConfig
from .data import escape_special_token_literals
from .model import AddSubLanguageModel
from .sft import SFT_FORMAT, SYSTEM_TEXT
from .train import CHECKPOINT_FORMAT, sha256_file


LEGACY_UNSCALED_CHECKPOINT_FORMATS = {
    "qmini-addsub-training-v4",
    "qmini-addsub-sft-v5",
    "qmini-addsub-training-v3",
    "qmini-addsub-sft-v4",
    "qmini-addsub-sft-v3",
    "qmini-addsub-sft-v2",
}
ACCEPTED_CHECKPOINT_FORMATS = {
    CHECKPOINT_FORMAT,
    SFT_FORMAT,
    *LEGACY_UNSCALED_CHECKPOINT_FORMATS,
}


def _special_ids(tokenizer: Tokenizer) -> dict[str, int]:
    tokens = (
        "<unk>", "<bos>", "<eos>", "<pad>", "<|system|>", "<|user|>",
        "<|assistant|>", "<|end|>",
    )
    values = {token: tokenizer.token_to_id(token) for token in tokens}
    missing = [token for token, token_id in values.items() if token_id is None]
    if missing:
        raise ValueError(f"Tokenizer is missing required control tokens: {missing}")
    normalized = {token: int(token_id) for token, token_id in values.items()}
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("Tokenizer control token IDs must be unique.")
    return normalized


def build_chat_prompt(
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    max_context: int,
    system_text: str = SYSTEM_TEXT,
) -> list[int]:
    """Encode chat turns, dropping old exchanges before trimming the current prompt."""
    if isinstance(max_context, bool) or not isinstance(max_context, int):
        raise TypeError("max_context must be an integer.")
    if max_context < 8:
        raise ValueError("max_context must be at least 8 tokens.")
    if not isinstance(system_text, str):
        raise TypeError("system_text must be a string.")
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a non-empty list or tuple.")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Invalid chat message at index {index}: expected an object.")
    if messages[0].get("role") != "user":
        raise ValueError("Chat history must start with a user message.")
    if messages[-1].get("role") != "user":
        raise ValueError("Chat history must end with a user message.")
    if messages[0].get("role") != "user":
        raise ValueError("Chat history must start with a user message.")

    control = _special_ids(tokenizer)
    segments: list[list[int]] = []
    previous_role = None
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"Invalid chat message at index {index}.")
        if role == previous_role:
            raise ValueError("Chat messages must alternate user and assistant roles.")
        encoded = tokenizer.encode(
            escape_special_token_literals(content), add_special_tokens=False
        ).ids
        if not encoded:
            raise ValueError(f"Chat message {index} is empty after tokenization.")
        role_id = control["<|user|>"] if role == "user" else control["<|assistant|>"]
        segments.append([role_id, *encoded, control["<|end|>"]])
        previous_role = role

    system_ids = tokenizer.encode(
        escape_special_token_literals(system_text), add_special_tokens=False
    ).ids

    def make_prefix() -> list[int]:
        return [
            control["<bos>"], control["<|system|>"],
            *system_ids, control["<|end|>"],
        ]

    def total_length() -> int:
        return len(make_prefix()) + sum(len(segment) for segment in segments) + 1

    # Remove only complete, oldest exchanges first.
    while total_length() > max_context and len(segments) > 1:
        del segments[:2]

    if total_length() > max_context:
        overflow = total_length() - max_context
        # Keep at least two system tokens when possible.
        removable_system = max(0, len(system_ids) - 2)
        trim_system = min(overflow, removable_system)
        if trim_system:
            system_ids = system_ids[trim_system:]

        overflow = total_length() - max_context
        if overflow > 0:
            last = segments[-1]
            content_ids = last[1:-1]
            available = max(0, len(content_ids) - overflow)
            if available < 1:
                raise ValueError(
                    "The current user message is too long for the configured context, "
                    "even after removing older history and shortening the system prompt."
                )
            tail = content_ids[-available:]
            segments[-1] = [last[0], *tail, last[-1]]

    prompt = make_prefix()
    for segment in segments:
        prompt.extend(segment)
    prompt.append(control["<|assistant|>"])
    if len(prompt) > max_context:
        raise ValueError("System prompt and current user message exceed the model context.")
    return prompt


def _apply_repetition_penalty(logits: torch.Tensor, generated: list[int],
                               penalty: float = 1.18, window: int = 64) -> torch.Tensor:
    """Apply a sign-aware penalty to recent generated token IDs."""
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not math.isfinite(penalty) or not 1.0 <= penalty <= 4.0:
        raise ValueError("repetition penalty must be finite and in [1, 4].")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("repetition window must be a positive integer.")
    scores=logits.float().clone()
    for token_id in set(generated[-window:]):
        if token_id<0 or token_id>=scores.numel(): raise ValueError("Generated token ID outside vocabulary.")
        scores[token_id]=scores[token_id]*penalty if scores[token_id]<0 else scores[token_id]/penalty
    return scores


def _would_exceed_fourgram_repeat(
    generated: list[int], next_token: int, max_repeats: int = 4
) -> bool:
    """Stop an exact four-token phrase before it would occur a fifth time."""
    if len(generated) < 3:
        return False
    candidate = (*generated[-3:], next_token)
    existing = sum(
        tuple(generated[index:index + 4]) == candidate
        for index in range(max(0, len(generated) - 3))
    )
    return existing >= max_repeats


def _choose_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
    generated: list[int] | None = None,
    repetition_penalty: float = 1.18,
    repetition_window: int = 64,
) -> int:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or logits.numel() < 1:
        raise ValueError("logits must be a non-empty one-dimensional tensor.")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits contain non-finite values.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer.")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be finite and in (0, 1].")
    scores = _apply_repetition_penalty(logits,generated or [],repetition_penalty,repetition_window)
    if greedy or temperature == 0:
        return int(torch.argmax(scores).item())

    scores = scores / temperature
    if top_k > 0 and top_k < scores.numel():
        keep_indices = torch.topk(scores, top_k).indices
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        keep_mask[keep_indices] = True
        scores = scores.masked_fill(~keep_mask, float("-inf"))

    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens whose inclusion would push cumulative mass over p,
        # but always retain the highest-probability token.
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[sorted_indices[remove]] = float("-inf")

    probabilities = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


@torch.inference_mode()
def generate_reply(
    model: AddSubLanguageModel,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    greedy: bool = False,
    system_text: str = SYSTEM_TEXT,
    repetition_penalty: float = 1.18,
    repetition_window: int = 64,
) -> str:
    """Generate an assistant turn using the model's cached PyTorch reference path."""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer.")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be finite and in (0, 1].")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative.")
    if isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, (int, float)) or not math.isfinite(repetition_penalty) or not 1.0 <= repetition_penalty <= 4.0:
        raise ValueError("repetition_penalty must be finite and in [1, 4].")
    if isinstance(repetition_window, bool) or not isinstance(repetition_window, int) or repetition_window < 1:
        raise ValueError("repetition_window must be a positive integer.")

    model.eval()
    device = next(model.parameters()).device
    prompt = build_chat_prompt(tokenizer, messages, model.config.max_context, system_text)
    generation_limit = min(max_new_tokens, model.config.max_context - len(prompt))
    if generation_limit < 1:
        raise ValueError("No context capacity remains for a generated assistant token.")

    control = _special_ids(tokenizer)
    stop_ids = {
        control["<unk>"], control["<eos>"], control["<|end|>"], control["<|user|>"],
        control["<|system|>"], control["<|assistant|>"], control["<bos>"],
        control["<pad>"],
    }

    caches = model.init_cache(batch_size=1, device=device)
    logits = None
    for token_id in prompt:
        current = torch.tensor([token_id], dtype=torch.long, device=device)
        logits, caches = model.forward_step(current, caches)
    if logits is None:
        raise RuntimeError("The encoded prompt is empty.")

    generated: list[int] = []
    for index in range(generation_limit):
        next_id = _choose_next_token(
            logits[0], temperature, top_k, top_p, greedy,
            generated=generated, repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        if next_id in stop_ids:
            break
        if _would_exceed_fourgram_repeat(generated, next_id):
            break
        generated.append(next_id)
        if index + 1 < generation_limit:
            current = torch.tensor([next_id], dtype=torch.long, device=device)
            logits, caches = model.forward_step(current, caches)

    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer or "I'm not sure how to answer that yet."


def load_addsub_model(
    checkpoint_path: str | Path,
    device_name: str = "cpu",
    tokenizer_path: str | Path | None = None,
) -> AddSubLanguageModel:
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in ACCEPTED_CHECKPOINT_FORMATS:
        raise ValueError(
            "Only recognized Q-Mini Add/Sub pretraining or SFT checkpoints are accepted."
        )
    if tokenizer_path is not None:
        tokenizer_file = Path(tokenizer_path)
        if not tokenizer_file.is_file():
            raise FileNotFoundError(tokenizer_file)
        expected_tokenizer_sha = checkpoint.get("tokenizer_sha256")
        actual_tokenizer_sha = sha256_file(tokenizer_file)
        if not expected_tokenizer_sha or actual_tokenizer_sha != expected_tokenizer_sha:
            raise ValueError(
                "Tokenizer fingerprint does not match the selected model checkpoint."
            )
    config = AddSubConfig.from_dict(checkpoint["config"])
    run_contract = checkpoint.get("run_contract", {})
    wavebit_options = (
        run_contract.get("wavebit_hsra") if isinstance(run_contract, dict) else None
    )
    model = AddSubLanguageModel(
        config, wavebit_options=wavebit_options
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    # The pre-scaled v1/v2 checkpoints used fixed, unscaled codebook thresholds.
    # Keep their reference behavior identical instead of silently applying the
    # new per-tensor dynamic scale to their old FP32 master weights.
    if checkpoint_format in LEGACY_UNSCALED_CHECKPOINT_FORMATS:
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter._qmini_fixed_scale = 1.0

    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with a trained Q-Mini Add/Sub v1 checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.18)
    parser.add_argument("--repetition-window", type=int, default=64)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = load_addsub_model(args.checkpoint, args.device, tokenizer_path=args.tokenizer)
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary {tokenizer.get_vocab_size()} does not match "
            f"checkpoint vocabulary {model.config.vocab_size}."
        )
    if not args.interactive and args.prompt is None:
        parser.error("Provide --prompt or use --interactive.")

    if args.interactive:
        history: list[dict[str, str]] = []
        print("Q-Mini Add/Sub reference chat. Type /exit to stop.")
        while True:
            try:
                user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_text.lower() in {"/exit", "/quit"}:
                break
            if not user_text:
                continue
            history.append({"role": "user", "content": user_text})
            answer = generate_reply(
                model, tokenizer, history,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                greedy=args.greedy,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
            )
            print("Q-Mini:", answer)
            history.append({"role": "assistant", "content": answer})
    else:
        answer = generate_reply(
            model, tokenizer,
            [{"role": "user", "content": args.prompt}],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            greedy=args.greedy,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
        )
        print(answer)


if __name__ == "__main__":
    main()
