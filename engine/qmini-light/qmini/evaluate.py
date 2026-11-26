from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .generate import generate
from .train import _load_masks, _load_tokens, _masked_cross_entropy, _sample_batch


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unicode_word_tokens(text: str) -> list[str]:
    """Tokenize Unicode letters/digits while retaining combining marks."""
    normalized = unicodedata.normalize("NFC", text.casefold())
    tokens: list[str] = []
    current: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if character.isalnum() or character == "_":
            current.append(character)
        elif category.startswith("M") and current:
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def normalize_answer(text: str) -> str:
    """Normalize Unicode word content for deterministic exact-match scoring."""
    if not isinstance(text, str):
        raise TypeError("Evaluation answers must be strings.")
    return " ".join(_unicode_word_tokens(text))


def token_f1(prediction: str, reference: str) -> float:
    """Multiset Unicode-word F1; lexical diagnostic, not a semantic score."""
    predicted = _unicode_word_tokens(prediction)
    expected = _unicode_word_tokens(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?"
)


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        ) / denominator
    )
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _parse_single_number(text: str) -> float:
    matches = _NUMBER_PATTERN.findall(text.replace(",", ""))
    if len(matches) != 1:
        raise ValueError("Expected exactly one numeric value.")
    value = float(matches[0])
    if not math.isfinite(value):
        raise ValueError("Numeric answer must be finite.")
    return value


def _score_task(
    prediction: str,
    reference: str,
    scorer: str,
    tolerance: float,
) -> tuple[bool, str | None]:
    """Task-specific scorers. Lexical F1 remains diagnostic, not semantic."""
    if scorer == "numeric":
        try:
            predicted_value = _parse_single_number(prediction)
            expected_value = _parse_single_number(reference)
        except ValueError as exc:
            return False, str(exc)
        return (
            math.isclose(
                predicted_value, expected_value,
                rel_tol=0.0, abs_tol=tolerance,
            ),
            None,
        )
    if scorer == "json":
        try:
            predicted_json = json.loads(prediction)
            expected_json = json.loads(reference)
        except json.JSONDecodeError as exc:
            return False, f"Invalid JSON: {exc.msg}"
        return predicted_json == expected_json, None
    if scorer == "contains":
        required = normalize_answer(reference)
        observed = normalize_answer(prediction)
        return bool(required) and required in observed, None
    if scorer == "lexical":
        return normalize_answer(prediction) == normalize_answer(reference), None
    raise ValueError(f"Unsupported task scorer: {scorer!r}")


def _task_records(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Task evaluation JSONL not found: {source}")
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid task JSON in {source} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Task record at {source}:{line_number} must be a JSON object.")
            prompt = record.get("prompt")
            reference = record.get("reference", record.get("expected", record.get("answer")))
            category = record.get("category", record.get("task", "uncategorized"))
            language = record.get("language", "unspecified")
            scorer = record.get("scorer", "lexical")
            tolerance = record.get("tolerance", 0.0)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Task record at {source}:{line_number} needs a non-empty prompt.")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(
                    f"Task record at {source}:{line_number} needs reference/expected/answer text."
                )
            if not isinstance(language, str) or not language.strip():
                raise ValueError(f"Task record at {source}:{line_number} has invalid language.")
            if scorer not in {"lexical", "numeric", "json", "contains"}:
                raise ValueError(
                    f"Task record at {source}:{line_number} uses unsupported scorer {scorer!r}."
                )
            try:
                tolerance = float(tolerance)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Task record at {source}:{line_number} has invalid numeric tolerance."
                ) from exc
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError(
                    f"Task record at {source}:{line_number} tolerance must be finite and non-negative."
                )
            yield {
                "prompt": prompt,
                "reference": reference,
                "category": str(category).strip() or "uncategorized",
                "language": language.strip().lower(),
                "scorer": str(scorer),
                "tolerance": tolerance,
            }


def evaluate_language_model(
    model,
    token_path: str | Path,
    mask_path: str | Path | None = None,
    batch_size: int = 2,
    seq_len: int = 512,
    batches: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate held-out next-token loss, throughput and per-layer expert load."""
    if min(batch_size, seq_len, batches) < 1:
        raise ValueError("batch_size, seq_len and batches must be positive.")
    if seq_len > model.config.max_context:
        raise ValueError("seq_len exceeds the model context window.")

    tokens = _load_tokens(token_path, "Evaluation")
    masks = _load_masks(mask_path, tokens, "Evaluation")
    if len(tokens) < seq_len + 1:
        raise ValueError(
            f"Evaluation stream needs at least {seq_len + 1} tokens; found {len(tokens)}."
        )

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    total_nll = 0.0
    target_tokens = 0
    input_tokens = 0
    router_totals: list[torch.Tensor] | None = None
    started = time.perf_counter()
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for _ in range(batches):
                inputs, targets, mask = _sample_batch(
                    tokens, masks, batch_size, seq_len, rng, device
                )
                output = model(inputs)
                loss = _masked_cross_entropy(output["logits"], targets, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Evaluation produced a non-finite loss.")
                count = int(mask.sum().item())
                total_nll += float(loss.item()) * count
                target_tokens += count
                input_tokens += int(inputs.numel())
                layer_counts = [
                    item.detach().to(device="cpu", dtype=torch.int64)
                    for item in output["router_counts"]
                ]
                if router_totals is None:
                    router_totals = layer_counts
                else:
                    for index, counts in enumerate(layer_counts):
                        router_totals[index] += counts
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        model.train(was_training)

    elapsed = max(time.perf_counter() - started, 1e-9)
    if target_tokens == 0:
        raise ValueError("Evaluation did not include supervised target tokens.")
    mean_loss = total_nll / target_tokens
    perplexity = math.exp(mean_loss) if mean_loss < 709.0 else None

    expert_report: list[dict[str, Any]] = []
    for layer_index, counts in enumerate(router_totals or []):
        values = [int(value) for value in counts.tolist()]
        total = sum(values)
        expert_report.append({
            "block": layer_index,
            "tokens_per_expert": values,
            "fraction_per_expert": [value / total for value in values] if total else [],
            "active_experts": sum(value > 0 for value in values),
        })

    return {
        "format": "qmini-language-evaluation-v1",
        "model_parameters": model.parameter_report()["total"],
        "evaluation_stream": str(Path(token_path)),
        "loss_mask": str(Path(mask_path)) if mask_path is not None else None,
        "seed": int(seed),
        "batches": int(batches),
        "batch_size": int(batch_size),
        "sequence_length": int(seq_len),
        "input_tokens": int(input_tokens),
        "supervised_target_tokens": int(target_tokens),
        "cross_entropy": float(mean_loss),
        "perplexity": float(perplexity) if perplexity is not None else None,
        "elapsed_seconds": float(elapsed),
        "input_tokens_per_second": float(input_tokens / elapsed),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        ),
        "router_load": expert_report,
    }


def evaluate_task_file(
    model,
    tokenizer,
    dataset_path: str | Path,
    max_new_tokens: int = 128,
    limit: int | None = None,
    thinking_steps: int | None = None,
) -> dict[str, Any]:
    """Evaluate a versioned task file with lexical and task-specific scoring."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when supplied.")

    was_training = model.training
    model.eval()
    total = 0
    exact = 0
    passed = 0
    f1_total = 0.0
    category_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0, "exact_match": 0, "passed": 0,
            "token_f1": 0.0, "scorers": set(),
        }
    )
    language_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "passed": 0}
    )
    cases: list[dict[str, Any]] = []
    try:
        for task in _task_records(dataset_path):
            if limit is not None and total >= limit:
                break
            prediction = generate(
                model, tokenizer, task["prompt"],
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                thinking_steps=thinking_steps,
            )
            lexical_exact = (
                normalize_answer(prediction) == normalize_answer(task["reference"])
            )
            score = token_f1(prediction, task["reference"])
            task_passed, scoring_error = _score_task(
                prediction,
                task["reference"],
                task["scorer"],
                task["tolerance"],
            )
            total += 1
            exact += int(lexical_exact)
            passed += int(task_passed)
            f1_total += score
            category = category_stats[task["category"]]
            category["count"] += 1
            category["exact_match"] += int(lexical_exact)
            category["passed"] += int(task_passed)
            category["token_f1"] += score
            category["scorers"].add(task["scorer"])
            language = language_stats[task["language"]]
            language["count"] += 1
            language["passed"] += int(task_passed)
            cases.append({
                "category": task["category"],
                "language": task["language"],
                "scorer": task["scorer"],
                "prompt": task["prompt"],
                "reference": task["reference"],
                "prediction": prediction,
                "passed": bool(task_passed),
                "scoring_error": scoring_error,
                "exact_match": bool(lexical_exact),
                "token_f1": float(score),
            })
    finally:
        model.train(was_training)

    if total == 0:
        raise ValueError("Task evaluation file contained no usable records.")
    categories = {}
    for name, values in sorted(category_stats.items()):
        count = int(values["count"])
        categories[name] = {
            "examples": count,
            "scorers": sorted(values["scorers"]),
            "task_pass_rate": values["passed"] / count,
            "task_pass_rate_wilson_95": _wilson_interval(
                int(values["passed"]), count
            ),
            "exact_match": values["exact_match"] / count,
            "mean_token_f1": values["token_f1"] / count,
        }
    languages = {}
    for name, values in sorted(language_stats.items()):
        count = int(values["count"])
        languages[name] = {
            "examples": count,
            "task_pass_rate": values["passed"] / count,
            "task_pass_rate_wilson_95": _wilson_interval(
                int(values["passed"]), count
            ),
        }
    return {
        "format": "qmini-task-evaluation-v2",
        "model_parameters": model.parameter_report()["total"],
        "dataset": str(Path(dataset_path)),
        "dataset_sha256": _sha256_file(dataset_path),
        "examples": total,
        "task_pass_rate": passed / total,
        "task_pass_rate_wilson_95": _wilson_interval(passed, total),
        "exact_match": exact / total,
        "mean_token_f1": f1_total / total,
        "categories": categories,
        "languages": languages,
        "cases": cases,
        "scoring_note": (
            "Numeric and JSON scorers perform structured checks; contains and lexical "
            "scorers are string-based diagnostics. None proves factuality, reasoning, "
            "code safety or general assistant capability. Code tasks require a separate "
            "constrained test harness and independently reviewed test cases."
        ),
    }

def emit_report(report: dict[str, Any], output_path: str | Path | None = None) -> None:
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if output_path is not None:
        target = Path(output_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.write("\n")
                handle.flush()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Evaluation report saved: {target}")
    print(serialized)



def _conversation_records(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Conversation evaluation JSONL not found: {source}")
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid conversation JSON in {source} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Conversation record at {source}:{line_number} must be an object."
                )
            turns = record.get("turns")
            if not isinstance(turns, list) or not turns:
                raise ValueError(
                    f"Conversation at {source}:{line_number} needs a non-empty turns array."
                )
            language = record.get("language", "unspecified")
            category = record.get("category", "conversation")
            system = record.get("system", "")
            if not isinstance(language, str) or not language.strip():
                raise ValueError(
                    f"Conversation at {source}:{line_number} has invalid language."
                )
            if not isinstance(category, str) or not category.strip():
                raise ValueError(
                    f"Conversation at {source}:{line_number} has invalid category."
                )
            if not isinstance(system, str):
                raise ValueError(
                    f"Conversation at {source}:{line_number} system prompt must be text."
                )
            normalized_turns = []
            for turn_index, turn in enumerate(turns, 1):
                if not isinstance(turn, dict):
                    raise ValueError(
                        f"Conversation turn {turn_index} at {source}:{line_number} must be an object."
                    )
                user_text = turn.get("user")
                reference = turn.get("reference", turn.get("expected", turn.get("answer")))
                scorer = turn.get("scorer", "lexical")
                tolerance = turn.get("tolerance", 0.0)
                turn_language = turn.get("language", language)
                turn_category = turn.get("category", category)
                if not isinstance(user_text, str) or not user_text.strip():
                    raise ValueError(
                        f"Conversation turn {turn_index} at {source}:{line_number} needs user text."
                    )
                if not isinstance(reference, str) or not reference.strip():
                    raise ValueError(
                        f"Conversation turn {turn_index} at {source}:{line_number} needs reference text."
                    )
                if scorer not in {"lexical", "numeric", "json", "contains"}:
                    raise ValueError(
                        f"Unsupported scorer {scorer!r} in conversation turn {turn_index}."
                    )
                if not isinstance(turn_language, str) or not turn_language.strip():
                    raise ValueError("Conversation turn language must be non-empty text.")
                if not isinstance(turn_category, str) or not turn_category.strip():
                    raise ValueError("Conversation turn category must be non-empty text.")
                try:
                    numeric_tolerance = float(tolerance)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Conversation turn tolerance must be numeric.") from exc
                if not math.isfinite(numeric_tolerance) or numeric_tolerance < 0:
                    raise ValueError("Conversation turn tolerance must be finite and non-negative.")
                normalized_turns.append({
                    "user": user_text,
                    "reference": reference,
                    "scorer": scorer,
                    "tolerance": numeric_tolerance,
                    "language": turn_language.strip().lower(),
                    "category": turn_category.strip(),
                })
            yield {
                "id": str(record.get("id", f"{source.name}:{line_number}")),
                "language": language.strip().lower(),
                "category": category.strip(),
                "system": system,
                "turns": normalized_turns,
            }


def evaluate_conversation_file(
    model,
    tokenizer,
    dataset_path: str | Path,
    max_new_tokens: int = 128,
    limit: int | None = None,
    thinking_steps: int | None = None,
) -> dict[str, Any]:
    """Evaluate multi-turn continuity using generated answers as later context."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when supplied.")

    required_tokens = ("<|system|>", "<|user|>", "<|assistant|>", "<|end|>")
    has_control_tokens = all(
        tokenizer.token_to_id(token) is not None for token in required_tokens
    )
    from .generate import _complete_chat_turns, trim_chat_history
    from .text import escape_special_token_literals

    was_training = model.training
    model.eval()
    conversations = 0
    exact = 0
    passed = 0
    f1_total = 0.0
    category_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0, "exact_match": 0, "passed": 0,
            "token_f1": 0.0, "scorers": set(),
        }
    )
    language_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "passed": 0}
    )
    cases: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for conversation in _conversation_records(dataset_path):
                if limit is not None and conversations >= limit:
                    break
                safe_system = escape_special_token_literals(conversation["system"])
                if has_control_tokens:
                    history = (
                        f"<|system|>{safe_system}\n<|end|>\n"
                        if safe_system else ""
                    )
                else:
                    if safe_system:
                        history = (
                            f"[SYSTEM]\n{safe_system}\n\n"
                        )
                    else:
                        history = ""

                for turn_index, turn in enumerate(conversation["turns"], 1):
                    _system_before, previous_turns = _complete_chat_turns(
                        history, has_control_tokens
                    )
                    safe_user = escape_special_token_literals(turn["user"])
                    user_block = (
                        f"<|user|>{safe_user}\n<|end|>\n<|assistant|>"
                        if has_control_tokens
                        else f"[USER]\n{safe_user}\n[ASSISTANT]\n"
                    )
                    prompt = history + user_block
                    if len(tokenizer.encode(prompt).ids) > model.config.max_context:
                        if not history:
                            raise ValueError(
                                f"Conversation {conversation['id']} turn {turn_index} "
                                "exceeds max_context by itself."
                            )
                        history = trim_chat_history(
                            history,
                            tokenizer,
                            token_budget=model.config.max_context,
                            has_control_tokens=has_control_tokens,
                            token_suffix=user_block,
                            allow_empty=True,
                        )
                        prompt = history + user_block
                        if len(tokenizer.encode(prompt).ids) > model.config.max_context:
                            raise ValueError(
                                f"Conversation {conversation['id']} turn {turn_index} "
                                "still exceeds max_context after whole-turn trimming."
                            )

                    _system_after, retained_turns = _complete_chat_turns(
                        history, has_control_tokens
                    )
                    dropped_history_turns = max(
                        0, len(previous_turns) - len(retained_turns)
                    ) if len(tokenizer.encode(prompt).ids) <= model.config.max_context else 0

                    prediction = generate(
                        model,
                        tokenizer,
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        top_k=0,
                        top_p=1.0,
                        thinking_steps=thinking_steps,
                        serialized_prompt=True,
                    )
                    lexical_exact = (
                        normalize_answer(prediction)
                        == normalize_answer(turn["reference"])
                    )
                    lexical_f1 = token_f1(prediction, turn["reference"])
                    task_passed, scoring_error = _score_task(
                        prediction,
                        turn["reference"],
                        turn["scorer"],
                        turn["tolerance"],
                    )
                    exact += int(lexical_exact)
                    passed += int(task_passed)
                    f1_total += lexical_f1

                    category = category_stats[turn["category"]]
                    category["count"] += 1
                    category["exact_match"] += int(lexical_exact)
                    category["passed"] += int(task_passed)
                    category["token_f1"] += lexical_f1
                    category["scorers"].add(turn["scorer"])
                    language = language_stats[turn["language"]]
                    language["count"] += 1
                    language["passed"] += int(task_passed)
                    cases.append({
                        "conversation_id": conversation["id"],
                        "turn_index": turn_index,
                        "category": turn["category"],
                        "language": turn["language"],
                        "scorer": turn["scorer"],
                        "user": turn["user"],
                        "reference": turn["reference"],
                        "prediction": prediction,
                        "passed": bool(task_passed),
                        "scoring_error": scoring_error,
                        "exact_match": bool(lexical_exact),
                        "token_f1": float(lexical_f1),
                        "history_tokens_before_turn": len(tokenizer.encode(history).ids),
                        "history_turns_retained_before_turn": len(retained_turns),
                        "history_turns_dropped_before_turn": dropped_history_turns,
                    })

                    safe_prediction = escape_special_token_literals(prediction)
                    if has_control_tokens:
                        history = prompt + safe_prediction + "\n<|end|>\n"
                    else:
                        history = prompt + safe_prediction + "\n"
                    conversations += int(turn_index == len(conversation["turns"]))

    finally:
        model.train(was_training)

    total_turns = len(cases)
    if total_turns == 0:
        raise ValueError("Conversation evaluation file contained no usable turns.")

    categories = {}
    for name, values in sorted(category_stats.items()):
        count = int(values["count"])
        categories[name] = {
            "turns": count,
            "scorers": sorted(values["scorers"]),
            "task_pass_rate": values["passed"] / count,
            "task_pass_rate_wilson_95": _wilson_interval(int(values["passed"]), count),
            "exact_match": values["exact_match"] / count,
            "mean_token_f1": values["token_f1"] / count,
        }
    languages = {}
    for name, values in sorted(language_stats.items()):
        count = int(values["count"])
        languages[name] = {
            "turns": count,
            "task_pass_rate": values["passed"] / count,
            "task_pass_rate_wilson_95": _wilson_interval(int(values["passed"]), count),
        }
    return {
        "format": "qmini-conversation-evaluation-v1",
        "model_parameters": model.parameter_report()["total"],
        "dataset": str(Path(dataset_path)),
        "dataset_sha256": _sha256_file(dataset_path),
        "conversations": conversations,
        "turns": total_turns,
        "task_pass_rate": passed / total_turns,
        "task_pass_rate_wilson_95": _wilson_interval(passed, total_turns),
        "exact_match": exact / total_turns,
        "mean_token_f1": f1_total / total_turns,
        "categories": categories,
        "languages": languages,
        "cases": cases,
        "history_policy": (
            "Generated assistant messages are reused as context. When necessary, only "
            "whole oldest turns are removed; role/end markers are preserved."
        ),
        "scoring_note": (
            "Turn-wise scoring is lexical or structured numeric/JSON checks. It does not "
            "prove factuality, safe behavior, reasoning, or code execution correctness."
        ),
    }
