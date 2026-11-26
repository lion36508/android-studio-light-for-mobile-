# Q-Mini Light 0.7M

A small, text-only language-model research repository. It currently contains **two model implementations**, not a family of trained 5M/150M/853M/5B models.

## Models in this repository

| Model implementation | Code | Current status |
|---|---|---|
| Q-Mini Light Legacy | `qmini/` | 705,742-parameter reference profile; Q2-v2 weights; preserve its existing checkpoints and tokenizer |
| Q-Mini Add/Sub | `qmini_addsub/` | 760,340-parameter experimental profile; independent training/data/tokenizer/checkpoint format; real assistant quality remains unverified |

These are separate code paths and must not share checkpoints, tokenizers, datasets or training outputs. They are still stored in one GitHub repository; the repository split is not yet complete.

## Legacy Q-Mini Light

The Light profile is `configs/qmini_light_0.7m.json`. It uses causal depthwise convolution, gated diagonal state-space memory, sparse Top-1 experts and shared refinement. Its production Q2-v2 weight alphabet is `{-1.0, -0.5, +0.5, +1.0}` (four symbols, two bits per symbol). The inference path includes multiplication/dot-product operations; it is not Add/Sub-only.

Legacy code, checkpoints and Drive data must be preserved during cleanup and migration.

## Q-Mini Add/Sub

The Add/Sub implementation is in `qmini_addsub/`. Its default configuration uses a 1,024-token vocabulary, width 96, four blocks, two Top-1 experts and a 512-token context. The model's five-symbol weight alphabet is `{-1.5, -0.5, 0, +0.5, +1.5}`, so fixed-width storage uses three bits per weight.

The training model uses PyTorch floating-point/autograd operations. The packed `.qma` archive is a verified weight container with float32 auxiliary tensors; it is **not** a complete integer-only inference engine. Optional WaveBit-HSRA is an experimental mode of this same Add/Sub model, disabled unless explicitly enabled. It is not a third model.

The Add/Sub data-preparation path uses hash-pinned Simple English Wikipedia text plus filtered OASST1 and human-authored Dolly instruction data. Review source licences before enabling downloads. User-selected local datasets can also be imported with provenance declarations.

## Supported Colab entry points

- `colab/QMini_English_CPU_One_Click_Resume.ipynb`: legacy Q-Mini Light training/resume.
- `colab/QMini_Existing_Drive_Data_CPU_Chat.ipynb`: chat using an existing legacy checkpoint.
- `colab/QMini_AddSub_V1_From_Scratch.ipynb`: independent Add/Sub experimental pipeline; data preparation, tokenizer build, training and evaluation switches are off by default.

See [Colab entry points](colab/README.md) and [Add/Sub implementation notes](qmini_addsub/README.md).

## Scope and non-claims

The active configuration set is limited to the legacy Light profile and the small CI smoke profile. Parameter count alone does not establish training progress or assistant capability.

Do not claim general reasoning, factuality, Android deployment, complete integer-only inference or speed gains without independent held-out evaluation and measurements on the target device. CI does not launch full training.

## Development

```bash
python -m pip install -e ".[dev]"
pytest -q
```

CI also compiles the native C++ projection primitive and cross-compiles its AArch64 implementation. Cross-compilation is not a real-device benchmark.
