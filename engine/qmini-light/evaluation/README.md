# Versioned evaluation smoke sets

These files are small, deterministic smoke inputs for exercising the Q-Mini evaluation code:

- smoke-tasks-v1.jsonl: 10 single-turn items covering English, Hindi and Assamese arithmetic, JSON formatting, simple instruction adherence and basic science.
- smoke-conversations-v1.jsonl: 3 short multi-turn continuity examples covering English, Hindi and Assamese recall.

They are not a statistically representative assistant benchmark, do not establish multilingual quality, and must not be used to select a model or claim general capability. The tasks are intentionally simple and exact; natural-language responses can fail lexical or substring scorers despite being semantically acceptable. Review case-level predictions.

Run the smoke sets against a Q-Mini checkpoint:

    python -m qmini.cli eval-tasks \
      --model-dir path/to/final_model \
      --dataset evaluation/smoke-tasks-v1.jsonl \
      --output reports/smoke-tasks.json

    python -m qmini.cli eval-conversations \
      --model-dir path/to/final_model \
      --dataset evaluation/smoke-conversations-v1.jsonl \
      --output reports/smoke-conversations.json

Reports include the exact SHA-256 of each JSONL file, per-category/per-language results, confidence intervals, and case-level predictions. Keep independent held-out datasets outside the training and SFT input folders. Freeze a dataset revision before model comparisons, and do not add these smoke files to any training data directory.

For full capability evaluation, curate separate and independently verified sets for multi-turn adherence, math, science/factual QA with sources, reasoning/state tracking, long-context recall, refusal/uncertainty, termination/repetition, and constrained code execution. Publish sample counts and scoring limitations per task.
