# Colab training boundary

The supported training workflows in this repository are the three notebooks indexed in [colab/README.md](../colab/README.md). They operate as separate legacy and Add/Sub paths.

The current Python trainers are single-process and single-device. This repository does not implement FSDP, DeepSpeed ZeRO, expert parallelism or tensor parallelism. Do not infer multi-GPU support from configuration estimates or hypothetical per-rank arithmetic.

The Add/Sub notebook defaults to CPU and leaves data preparation, tokenizer construction, pretraining, SFT and evaluation disabled. Enable stages only after checking source licences, free storage, available memory and the exact dataset/tokenizer fingerprints. The legacy notebook uses its own existing data/checkpoint locations; do not point it at Add/Sub files.

CI does not execute full Colab training. It tests bounded synthetic code paths only. No GPU throughput, model-quality gain or device compatibility claim should be inferred from these tests.
