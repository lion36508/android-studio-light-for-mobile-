# Strict 2-bit migration: verified baseline and implementation gates

**Status:** migration not complete. This document records the verified source-level blockers and the order in which they must be removed. It does not relabel the existing Q2-v2 archive as fully 2-bit.

## Current baseline

The current deployment format is mixed precision:

- Matrix weight symbols are genuinely packed at two bits per weight.
- Each Q2 matrix row still carries an FP32 scale.
- One-dimensional parameters are stored as FP16, not two-bit values.
- The native projection path uses INT8 activations, an INT64 integer accumulator, and FP32 scales/bias/output.

The current trainer also initializes QMiniMoE(config) using PyTorch's default floating parameter dtype. Its Q2 fake-quantization uses a straight-through estimator over those trainable master parameters. torch.autocast changes eligible forward operations; it does not convert the model's stored parameters to two-bit values. AdamW creates optimizer state from the trainable parameters, so the current FP32 parameter path also creates FP32 optimizer state.

Relevant source locations:

- qmini/train.py: model construction, AdamW, autocast, resume checkpoint and run/final manifests.
- qmini/quantization.py: Q2 packing/dequantization, FP32 scales and FP32 conversion helpers.
- qmini/export.py: Q2 matrix archive plus FP32 row scales and FP16 vectors.
- qmini/packed_runtime.py: validation/dequantization and loading into a floating trainable model.
- qmini/model.py: FP32-sensitive normalization, SSM recurrence, router and auxiliary-loss operations.
- qmini/native.py and qmini/native_kernel.cpp: FP32 row/activation scales, bias and output around integer accumulation.

## Required migration sequence

1. **Inventory first.** Run python scripts/precision_audit.py --output precision-audit.json. Review every occurrence. Static matches are leads, not proof: distinguish persistent weights/state from temporary arithmetic and diagnostic references.
2. **Version the artifact format.** A strict candidate must define how all trainable tensors—including vectors, biases and normalization parameters—and row-scale metadata are encoded. Do not silently change Q2-v2 or break its reader. Keep v1/v2 readers and add round-trip, corruption and compatibility tests for any new format.
3. **Replace the optimizer algorithm.** Removing FP32 master parameters is not achieved by changing a dtype string. A genuinely two-bit persistent training state needs a specified update rule, gradient/accumulator representation, saturation policy, checkpoint/resume format and deterministic test oracle. The existing STE + AdamW implementation is not that algorithm.
4. **Separate storage from arithmetic.** State exactly which precision is used for multiply, accumulation, normalization, SSM recurrence, loss and reductions. Integer accumulation can coexist with low-bit weights; it does not make the entire training computation two-bit. Any non-two-bit temporary must be visible in the manifest and measured.
5. **Prove quality and efficiency.** Compare against the same random-initialized Q-Mini checkpoint and data split. Required evidence includes held-out NLL, gradient/update stability, task metrics, save/resume parity, peak RAM/VRAM, tokens/sec, artifact bytes and ARM64 runtime tests where available.
6. **Fail closed.** Once each migration stage is implemented, add CI gates that reject its prohibited persistent dtype/storage. Do not enable a strict gate before the corresponding implementation and tests exist.

## Release rule

Do not claim "2-bit only" while FP32 row scales, FP16 vectors, FP32 master parameters or FP32 optimizer state remain in the relevant path. Do not delete previous checkpoints or Drive artifacts during migration. Build versioned candidates alongside existing artifacts and promote only after matched tests pass.
