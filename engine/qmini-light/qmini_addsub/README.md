# Q-Mini Add/Sub

Q-Mini Add/Sub is the repository's **one experimental new-model implementation**. It is not a separate third-party model and it must not load legacy Q2-v2 checkpoints.

## Default profile

- 1,024-token vocabulary, width 96, four blocks, two Top-1 experts.
- 512-token context and 760,340 parameters.
- Causal depthwise convolution, gated diagonal state-space memory, sparse Top-1 SwiGLU experts, RMSNorm and residual branches.
- Five quantized weight symbols: `{-1.5, -0.5, 0, +0.5, +1.5}`; fixed-width packing requires 3 bits per weight.

This is an experimental model size, not a quality score. The public repository does not establish that this model has learned useful general-purpose conversation.

## OxCore experimental native core

A standalone native CPU primitive layer for this same Add/Sub implementation
lives under `qmini_addsub/oxcore/`. It provides a C++20 ABI, a caller-owned
fixed-capacity arena, bounded tensor views, Mirror-63 packing and integer
projection/convolution, a Q15 diagonal SSM scan, integer FWHT, Sinkhorn-4 and
L2 normalization. Build and run its native sanitizer checks with:

```bash
bash scripts/build_oxcore.sh /tmp/libqmini_oxcore.so
bash scripts/test_oxcore_sanitizers.sh
```

OxCore is not yet wired into the Add/Sub model forward pass, trainer,
checkpoint or export path. It does not replace the PyTorch reference graph,
and it is not a complete integer-only model runtime. See
[`oxcore/README.md`](oxcore/README.md) and
[`oxcore/MATH_AUDIT.md`](oxcore/MATH_AUDIT.md) for scope and deferred work.

## Training and data

The implementation has its own tokenizer, data manifest, pretraining checkpoints, SFT checkpoints, evaluation and export path. The standard training model uses PyTorch floating-point operations and autograd.

The optional pinned data-preparation command uses:
- Simple English Wikipedia article text for pretraining (CC-BY-SA-3.0).
- Filtered English OpenAssistant OASST1 conversations (Apache-2.0).
- Human-authored Databricks Dolly instructions (CC-BY-SA-3.0).

The script pins source revisions and hashes and writes the dataset transactionally. It requires `--accept-source-licenses` after the source terms have been reviewed. Install the declared optional dependencies with:

```bash
python -m pip install -e ".[addsub-data]"
python scripts/prepare_qmini_addsub_data.py --accept-source-licenses
```

A user-selected local corpus can be imported instead. The import manifest records source/license declarations and hashes; those declarations are not an independent legal or authorship audit. Train the tokenizer only after checking the train/validation/test split and provenance.

The Colab notebook `colab/QMini_AddSub_V1_From_Scratch.ipynb` is the supported interactive entry point. All data and training stages are disabled by default.

## Checkpoints and generation

Add/Sub accepts only its own checkpoint formats and verifies the tokenizer/data fingerprints before resuming or fine-tuning. Keep each dataset, tokenizer, run directory and checkpoint lineage together. Never point it at a legacy `qmini/` checkpoint or legacy Drive folder.

The PyTorch generation path is a correctness/reference path. The `.qma` export packs five-symbol weights and verifies archive hashes, but its auxiliary tensors remain float32 and its loader reconstructs the PyTorch reference model. The complete integer-only convolution/state-space/MoE/normalization/output graph and production ARM64 runtime are not complete.

## Optional WaveBit-HSRA

WaveBit-HSRA is an opt-in experimental mode inside this same Add/Sub implementation. It is off by default and is not a separately trained model. Its native projection primitives and archive format are not proof of full-model multiplication-free inference, higher quality or a speedup. No such claims should be made without complete end-to-end parity and matched held-out measurements.

## Evaluation limits

The evaluator records generated/reference pairs and lexical metrics such as exact match and token overlap. These do not measure semantic correctness, factuality, reasoning or safety. A passing software test suite is not proof of conversational quality. Real-data pretraining/SFT and independent held-out evaluation are still required before describing Add/Sub as a capable assistant.
