# WaveBit-HSRA integration in Q-Mini's existing new model

WaveBit-HSRA is an opt-in mode of the existing zero-inclusive
qmini_addsub.AddSubLanguageModel. It is not a third model or a separate
architecture/checkpoint family. The original zero-free model track and its
files are not modified.

## What is integrated

- Group-wise scales are computed in configurable input groups (default 30
  weights per output row); no trainable scale parameters are added.
- Octave-Shift uses base levels {-1, -0.5, 0, +0.5, +1}. Each triplet shares
  one mode bit. The default boost factor remains x4 for compatibility, while
  x2 is available as an explicit ablation.
- One triplet occupies one byte before zero-run compression: three base-5
  symbols plus a shared boost bit (2.667 bits/weight, before group scales,
  offsets and archive metadata).
- Codes 250..254 skip 4/8/16/32/64 all-zero triplet-groups; 255 is followed by
  an extended skip count. Skip units are groups of three weights.
- Bounded EWGS-style gradients are used by the WaveBit training surrogate.
- Each existing Add/Sub block gets a parameter-free fixed-width holographic
  rotation state. Its per-token context modulates the existing linear
  projections, depthwise convolution, SSM projections, MoE router/experts and
  tied output projection. The input embedding also uses the group-wise
  quantizer; the output head remains tied to the same embedding parameter.
- Incremental generation carries the holographic state and convolution
  context history in the existing cache path.
- The fixed blockwise Hadamard transform and activation fake quantization are
  applied to the existing expert down-projections when enabled. The default
  expert hidden width 170 is kept unchanged (21 blocks of 8 plus a 2-wide
  power-of-two tail).
- The optional CapacitorCharge integer helper is included for a later
  optimizer experiment, but is deliberately not enabled in the default
  optimizer. Its learning effect has not yet been established.

## Parameter/checkpoint compatibility

WaveBit memories, group scales and Hadamard transforms add no trainable
parameters and do not create new state_dict keys. Old checkpoints with no
run_contract.wavebit_hsra load using the original Add/Sub behavior. New
WaveBit runs record their settings in the training run contract; SFT and
generation reconstruct the same mode from that metadata.

## CPU training and group-wise export

Example CPU pretraining invocation (after the verified English dataset and
tokenizer have been prepared):

    python -m qmini_addsub.train \
      --train-bin data/train.bin \
      --valid-bin data/validation.bin \
      --tokenizer data/tokenizer.json \
      --output runs/qmini_addsub_x2_hadamard \
      --device cpu \
      --wavebit-hsra \
      --wavebit-group-size 30 \
      --wavebit-octave-boost-factor 2 \
      --wavebit-hadamard \
      --wavebit-hadamard-block-size 8 \
      --wavebit-activation-bits 4

Export a trained WaveBit checkpoint with the same command as the legacy
Add/Sub exporter:

    python -m qmini_addsub.export \
      --checkpoint runs/qmini_addsub_x2_hadamard/checkpoint-step-000001000.pt \
      --output exports/qmini-addsub-wavebit-x2.qma

WaveBit checkpoints now use a separate, versioned archive format:
qmini-addsub-wavebit-packed-v1. It contains manifest.json, weights.bin,
auxiliary.npz and group_metadata.npz. The group metadata stores float32 scales
for each output-row/input-group and row offsets for the variable-length
triplet stream. The manifest fingerprints each payload and records the boost
factor, group size, scale layout, skip grammar, tokenizer and dataset hashes.
The loader verifies these fields, reconstructs the exact parameter shapes,
and freezes the saved group scales in the PyTorch reference model so a loaded
checkpoint does not silently recalculate different scales.

The existing qmini-addsub-packed-weights-v1/v2 formats are unchanged. The
WaveBit archive is a packed-weight container plus a reconstructable reference
model; it is not a complete integer-only inference archive.

## Native integer primitives

- qmini_octave_linear_i8_v1 remains the stable x4 ABI.
  qmini_octave_linear_i8_v2 takes an explicit shared x2 or x4 boost factor.
  Both decode the triplet stream and return integer half-unit accumulators.
  The Python wrapper routes x4 to v1 when available and x2 to v2.
- qmini_hadamard_quantize_i8_v1 applies the blockwise Walsh-Hadamard
  butterfly to int8 activation codes using integer add/sub operations, then
  produces symmetric 2..8-bit activation codes and one output scale per row.
  The Python NativeKernel wrapper and NumPy integer reference are covered by
  parity tests, including a 170-feature width and non-power-of-two tails.
- The Hadamard primitive performs floating-point normalization/requantization
  for the activation scale. Therefore it is not a claim of a fully
  multiplication-free end-to-end inference path.

## Group-wise native Octave projection

The optional qmini_octave_groupwise_linear_i8_v1 ABI and
NativeKernel.octave_groupwise_linear wrapper consume the packed triplet
stream, per-output-row/per-input-group float32 scales, one float32 activation
scale per input row, and an optional float32 bias. They validate the scale
matrix against ceil(in_features / group_size), including group sizes that do
not align to a three-weight triplet.

Each input group is accumulated separately in integer half-units using the
Octave shift/add coefficient path. The final projection applies that group's
weight scale and the input-row activation scale, then adds bias. This closes
the earlier scale-layout gap for a standalone native linear projection.
The older qmini_octave_linear_i8_v1/v2 half-unit ABI remains unchanged.

The dot-product accumulation itself uses integer add/sub/left-shift operations;
however, the final group-scale and activation-scale composition is FP32 and
uses multiplication. The new API is therefore scale-aware, not a claim of a
fully multiplication-free inference runtime. The grouped accumulator
uses the shared 8-lane NEON shift/add routine where the target supports it,
with scalar handling for group tails and non-NEON targets. ARM64 cross-
compilation verifies build compatibility, not NEON execution or target-device
speed; no groupwise speedup is claimed without a real-device benchmark. The
API is also not yet connected to all Add/Sub model layers or the generation
loop.

## Explicit limitations

- PyTorch training still uses floating-point autograd and dense operators.
  This integration is not a zero-float runtime and makes no speed claim.
- The native Hadamard and Octave projection functions are separate primitives.
  The Hadamard output is not yet wired automatically into every model
  projection in qmini_addsub generation.
- The legacy Octave projection ABI still returns raw half-unit accumulators.
  The new group-wise ABI applies per-group weight scales, per-row activation
  scales and bias for a standalone linear projection. Normalization, gates,
  SSM, MoE routing, embedding/head integration and output requantization still
  need to be composed into a complete generation runtime.
- The archive loader reconstructs a PyTorch reference model; it does not
  execute the full model directly from packed bytes.
- The C++ ABI includes an 8-lane ARM NEON shift/add path for the Octave
  projection and a scalar fallback. The Hadamard preprocessing primitive is
  currently a scalar integer butterfly; target-device optimization and
  measurements remain future work.
- This is a small L1-friendly software lookup table, not a single ARM NEON
  TBL instruction; the NEON TBL table operand is limited to four 16-byte vectors.
- The capacitor-charge helper is not active in AdamW training; it must be
  evaluated as a separate optimizer ablation before activation.
- One shared mode bit cannot independently boost one weight within a triplet;
  independent nine-level choices for three weights require at least 10 bits.
- Skip savings occur only for clustered all-zero triplet-groups and are not
  guaranteed for dense/non-zero weight distributions.
- Held-out English quality, latency, energy and memory comparisons against
  ternary and previous Add/Sub baselines have not yet been completed. No
  accuracy or speed improvement is claimed from software tests alone.

The existing Add/Sub model remains the single zero-inclusive evolving model.
All WaveBit additions are implemented inside that model path, not alongside it.
