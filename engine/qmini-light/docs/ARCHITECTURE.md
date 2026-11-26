# Q-Mini architecture

This repository contains two implementations. They share broad research ideas but have separate code, configurations, tokenizers, checkpoint formats and training pipelines.

## Legacy Q-Mini Light

Source: `qmini/`  
Active configuration: `configs/qmini_light_0.7m.json`  
Documented reference size: 705,742 parameters.

The forward path is text-only and attention-free: causal depthwise convolution, gated content/control projection, a gated diagonal state-space path, sparse Top-1 expert routing, shared refinement and a tied output projection. The production matrix format is Q2-v2 with four symbols `{-1.0, -0.5, +0.5, +1.0}`. It uses two bits per weight symbol and has no exact-zero symbol for an individual weight in a non-zero row.

The legacy inference path uses multiplication/dot products and convolution. It must not be described as an Add/Sub-only engine.

## Q-Mini Add/Sub

Source: `qmini_addsub/`  
Default reference configuration: 760,340 parameters, width 96, four blocks, two experts, 1,024 vocabulary and 512-token context.

It uses causal depthwise convolution, gated diagonal state-space memory, sparse Top-1 experts, RMSNorm and residual branches. Its training code is independent of the legacy package. Its five-symbol weight alphabet is `{-1.5, -0.5, 0, +0.5, +1.5}`; five symbols require at least three bits in fixed-width storage.

The PyTorch training/reference path uses floating-point operations. The packed archive stores quantized weights plus float32 auxiliary tensors. The current integer reference primitives do not constitute a complete integer-only model graph.

Optional WaveBit-HSRA is an opt-in experimental mode inside Add/Sub, disabled by default. It does not create another model family and has no verified end-to-end speed or quality claim.

## Isolation rules

- Never load a legacy Q2-v2 checkpoint into Add/Sub.
- Never reuse a tokenizer, dataset manifest, optimizer state or checkpoint across the two implementations.
- Keep Drive roots and output directories separate.
- Preserve legacy artifacts during any repository migration.
- Treat test fixtures and random-initialization smoke runs as software verification, not trained-model evidence.

## Runtime boundary

The legacy packed runtime and the Add/Sub reference runtime have different arithmetic contracts. Neither the current Add/Sub archive nor ARM64 cross-compilation proves a complete integer-only Android model. Real-device memory, speed and thermal behavior require measurements on the actual device.
