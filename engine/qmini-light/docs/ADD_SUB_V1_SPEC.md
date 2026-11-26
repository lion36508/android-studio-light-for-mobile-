# Q-Mini Add/Sub v1 — Architecture Contract

**Status:** first independent training/data implementation exists; no Add/Sub model has been trained and the full multiply-free inference graph is not yet complete.

## 1. Product-line separation

| Legacy line | New line |
|---|---|
| Python package: `qmini/` | Package boundary: `qmini_addsub/` |
| Q2-v2 artifact | New artifact format: `qmini-addsub-v1` |
| 4,096 vocabulary legacy profile | Independently versioned config/tokenizer |
| Existing Q-Mini Light checkpoints | New checkpoints only |
| Existing Drive folder | Separate `QMini-Light-AddSub-v1/` Drive folder |

The old model remains runnable and its artifacts remain untouched. The new code must not import legacy model/runtime classes or load legacy weights. Stable ideas such as data manifests, checksums, tests and resumable-save principles may be reused, but model, quantization, tokenizer, checkpoint and runtime code stay separate.

## 2. Required weight alphabet

The new model's discrete weight alphabet is exactly:

`{-1.5, -0.5, 0, +0.5, +1.5}`

Zero is a real symbol. A zero weight means no connection contribution and must be skipped exactly; it must not be approximated as a small positive or negative weight.

**Storage fact:** five distinct symbols cannot fit in a fixed 2-bit code (2 bits provide only four codewords). The initial packed format therefore requires at least 3 bits per weight, with three unused 3-bit codes reserved for future versioning or rejected as invalid. Do not market this alphabet as 2-bit.

## 3. Inference arithmetic rule

The production inference path must use only:

- integer addition
- integer subtraction
- comparisons and index selection
- constant-time table lookup for coefficient contributions and nonlinear functions

No runtime multiplication, division, floating-point rescaling, matrix multiplication, dot product, tensor `@`, `torch.matmul`, `F.linear`, convolution multiply, or fused multiply-add is permitted.

A coefficient lookup maps the activation integer and one of the five weight symbols to a precomputed integer contribution. The inference accumulator then combines contributions only with add/subtract. The zero symbol maps to a skipped contribution. Activation functions, normalization and other nonlinear transforms require fixed lookup tables or an alternative explicitly approved add/sub-only implementation. They must not hide multiplication in a scale, gate or normalization step.

This requirement covers the *entire deployed graph*, not only the projection kernel.

## 4. Model architecture and training graph

The independent Add/Sub v1 training graph currently reuses three useful architectural ideas from the legacy family, but with new code and a reduced, separately versioned configuration:

- causal depthwise convolution for local sequence mixing;
- gated diagonal state-space recurrence for running context;
- sparse Top-1 expert routing with per-token selected expert computation.

The default configuration is English-first with a 1,024-token vocabulary, width 96, four blocks, two experts and a 170-wide SwiGLU bottleneck per expert. Each block uses RMSNorm, causal depthwise convolution with dilations 1/2/4/8, an input-dependent gated diagonal state-space recurrence with an associative parallel scan, and sparse Top-1 expert dispatch. Three learnable sigmoid-bounded residual branch gates control the local convolution, state-space and expert updates; the v2 training algorithm initializes each gate at 0.1 so branches begin as controlled residual corrections rather than 0.5-strength updates. Sparse routing uses a Switch-style load-balancing objective plus a separately weighted log-partition-squared router z-loss; both objectives use only valid non-padding tokens. Per-layer assignment fractions and router losses are logged so expert collapse is observable. A hard activation clamp is intentionally avoided so gradients are not cut at a clipping boundary. The current instantiated training model has **760,340 parameters**. This is an experimental design, not a claim of lower validation loss until measured on held-out English data.

The training graph uses five-symbol STE/QAT matrix weights and sparse SwiGLU experts. Pretraining and SFT use AdamW with matrix-weight decay, no decay on vector/bias/norm parameters, linear LR warm-up and cosine decay to a 10% floor. The MoE balance and router z-loss coefficients are explicit configuration values stored in run/checkpoint identity. Training/optimization is allowed to use floating-point arithmetic, multiplication and gradient operations. The deployment restriction applies to the inference graph only. Do not call the training model the Add/Sub-only runtime.

## 5. Training policy

Training precision is an independent research choice. This contract does **not** require a staged FP16/BF16 phase, does not prescribe an FP16-to-low-bit schedule, and does not authorize any change to the legacy model's trainer.

The new model is initialized independently and trained only using the project's approved, provenance-checked English corpus. Do not load external pretrained checkpoints, adapters, LoRA weights or old Q-Mini weights.

## 6. Required package layout

```text
qmini/                         # existing legacy Q-Mini Light; preserve
qmini_addsub/
    README.md
    config.py                  # independent model configuration
    model.py                   # add/sub graph only
    lookup_tables.py           # immutable, versioned contribution/activation tables
    arithmetic.py              # bounded integer add/sub accumulation
    tokenizer.py               # independent tokenizer contract/version
    data.py                    # provenance and document-level split
    train.py                   # new training implementation
    checkpoint.py              # new checkpoint format; never reads qmini files
    export.py                  # offline five-symbol weight container and validation
    runtime.py                 # integer projection / residual primitives; full graph pending
    tests/
```

Implemented files include config.py, model.py, quantization.py, runtime.py, data.py, train.py and sft.py. Data/tokenizer preparation scripts and tests live in scripts/ and tests/. A cached PyTorch streaming reference path now exists for parity-tested one-token generation. It is not the integer-only deployed runtime. The offline five-symbol checkpoint weight-container exporter and reconstruction loader now exist, with SHA-256 checks, shape/index validation and tamper tests. This export is not an inference runtime: auxiliary tensors remain float32. Calibrated LUTs and the full integer-only convolution/SSM/MoE/normalization/output graph plus native device runtime are still pending. The legacy package must not import from qmini_addsub, and the new package must not import model/runtime classes from qmini.

## 7. Artifact compatibility

The manifest must include:

- `format = qmini-addsub-v1`
- exact alphabet and code mapping, including zero
- bits per weight and rejected/reserved codewords
- accumulator width, signedness and saturation/overflow policy
- lookup-table hashes and versions
- tokenizer hash and vocabulary size
- config hash and actual parameter count
- source dataset manifest hash and training-run identity
- proof/test report identifying the audited inference runtime build

The loader must reject Q2-v1, Q2-v2, dense PyTorch checkpoints and any unknown artifact format.

## 8. Acceptance tests

1. Round-trip all five symbols, including exact zero, across pack/unpack.
2. Invalid 3-bit codes are rejected; no symbol silently maps to another.
3. Exhaustive small-domain parity against a simple add/sub reference interpreter.
4. Tests prove zero weights add no contribution.
5. Static and compiled-binary audit rejects multiply/divide/matmul/FMA instructions in the deployed inference graph.
6. Test all nonlinear and normalization paths; no hidden multiply is allowed.
7. Same seed/config/data produces deterministic initialization and reproducible export.
8. Save/resume parity and packed-weight export tamper detection pass.
9. Held-out English NLL/BPB and assistant-response test evaluation are reported separately from synthetic smoke tests. The initial lexical metrics do not establish semantic correctness.
10. Real target-device RAM, speed, thermal and energy measurements are published before phone-support claims.

## 9. Explicit non-goals for v1

- Do not merge with legacy Q2-v2 weights or checkpoints.
- Do not overwrite or migrate the old Drive dataset/checkpoint folders.
- Do not re-enable TinyStories or teacher-generated synthetic samples as production training data.
- Do not claim 2-bit storage for five symbols.
- Do not claim an add/sub-only runtime until the complete deployed graph and compiled instructions pass the acceptance tests.


## 10. 23 November 2026 — scaled QAT and tokenizer revision

- Fixed thresholds created a broad forward quantization deadzone in master-weight units. The STE still passed gradients; the problem was that small master-weight updates did not change forward symbols.
- The revised training graph uses dense Kaiming-normal master weights and a detached per-tensor scale (1.4 × mean absolute weight) before selecting five normalized symbols. Exact zero remains a code; zero-initialized SSM decay weights still receive STE gradients.
- Matrix weight decay is 0.01. Pretraining and SFT checkpoint IDs were bumped to prevent incompatible resumes.
- The tokenizer target is 1,024. The full reversible 256-byte ByteLevel fallback remains; a maximum 64 MiB sample of pretraining training text is balanced with the complete SFT training split. Validation/test are excluded.
- WikiText `@-@` and spaces before punctuation are normalized. System prompt is `You are Q-Mini.`; reference decoding uses configurable repetition penalty, default 1.18 over recent 64 tokens.
- Packed archive v2 records one scale per tensor and keeps v1 read compatibility. It is not a complete integer-only Android runtime.
- New model has 760,340 parameters; old 710,676-parameter checkpoints remain preserved but require fresh training under the new tokenizer and optimizer contract.


## 11. 23 November 2026 — Simple English pretraining source

The default pinned pretraining source is the human-authored Simple English Wikipedia snapshot from 2026-03-01. It has 276,302 articles in the published card; the 439,202,943-byte parquet file is pinned by revision and SHA-256. Only `title` and `content_no_link` are used. Generated importance labels are not training targets or text. Stable page-ID hashing makes 90/5/5 train/validation/test splits. The source is CC-BY-SA-3.0 and requires attribution/share-alike review. The old WikiText helper remains available in source for historical compatibility but is no longer the default pinned pretraining corpus.
