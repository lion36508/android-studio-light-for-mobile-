# Q-Mini Light project scope

## Purpose

Maintain a small, text-only language-model research project with two independently implemented model lines: the legacy Q-Mini Light baseline and the experimental Q-Mini Add/Sub model. The project is not a Transformer, image model, multimodal model or third-party chatbot wrapper.

## Current implementations

| Implementation | Source | Configuration | Status |
|---|---|---|---|
| Q-Mini Light Legacy | `qmini/` | `configs/qmini_light_0.7m.json` | Preserved legacy baseline; keep its Q2-v2 checkpoints and tokenizer |
| Q-Mini Add/Sub | `qmini_addsub/` | Add/Sub default config | Experimental independent training path; full integer-only inference is incomplete |

The implementations must not share weights, tokenizers, optimizer state, datasets or training outputs. They remain in one GitHub repository at present; separate repository migration is a distinct task.

## Architecture constraints

- Text-only causal sequence model.
- No Transformer self-attention or Q/K/V attention layers.
- Legacy production weights use the four-symbol Q2-v2 codebook.
- Add/Sub uses a separate five-symbol codebook and three-bit packed format.
- Optional WaveBit-HSRA remains experimental and disabled unless explicitly selected.
- Any new architecture must be introduced as a separately tested implementation, not as an unsupported capability claim.

## Out of scope

The active scope is limited to the two implementations listed above. Distributed multi-device training, complete integer-only Add/Sub inference and production Android deployment are not delivered features.

## Evidence standard

A feature is called implemented only when code and tests exist. Training capability requires reproducible training on approved data. Assistant quality requires independent held-out tasks, multi-turn evaluation and human review. Performance claims require actual measurements on named hardware. Unit tests and synthetic smoke reports are not language-quality benchmarks.
