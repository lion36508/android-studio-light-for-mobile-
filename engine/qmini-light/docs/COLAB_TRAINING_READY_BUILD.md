# Q-Mini legacy Colab build and training lifecycle (historical)

> **Status:** This document describes the legacy Q2-v2 workflow. It is retained for history and regression context, not as the current Add/Sub training guide. Its notebook snapshots are read-only fixtures under `../tests/fixtures/legacy_colab/`. Use `../colab/README.md` for the supported entry points.

## Stage A — build the model package (training stays off)

Open [QMini Training-Ready Model Builder](../tests/fixtures/legacy_colab/QMini_Training_Ready_Build.ipynb).

Place reviewed corpus files in `MyDrive/QMini-Light-0.7M/datasets/`. The notebook prepares the byte-level BPE tokenizer, train/validation token streams and loss masks, then builds the random-initialized Q2-v2 model. It bundles the current `qmini/` source tree—including the training loop and checkpoint manager—under `training_runtime/`.

The model builder has no training-enable switch and does not construct an optimizer, run backward passes or update model weights. The package starts with:

- `initial_brain/model.qmini.npz`, `config.json`, and `model_manifest.json`
- `tokenizer/tokenizer.json` and its provenance sidecar
- `dataset/train.bin`, `valid.bin`, both loss masks and `dataset.json`
- `training_runtime/qmini/` — the bundled training/runtime implementation
- `training_runtime/TRAINING_RUNTIME.json` — source hashes and `enabled: false`
- `training_state/training_progress.json` — the one current progress record, initially step 0
- `BUILD_REPORT.json` — model, data, tokenizer, precision and build evidence

The package is stored at `MyDrive/QMini-Light-0.7M/training_ready/QMini-Light-0.7M-Q2v2-<fingerprint>/`.

## Stage B — start/resume training separately

Open [QMini Training Runner](../tests/fixtures/legacy_colab/QMini_Training_Runner.ipynb). It imports the exact training code bundled with the selected package and checks the source-file hashes, model manifest, tokenizer, dataset, Q2 archive and GPU memory.

Training is off by default. Set `START_TRAINING = True` only when ready. The runner creates its working checkpoint on Colab's local disk and uses the Drive folder `training_state/` as the durable checkpoint mirror.

The runner's default training profile is the repository's 0.7M-class hybrid configuration, with a 10-step checkpoint interval. This bounds the amount of training that must be repeated after a sudden interruption; use a smaller interval only after considering the additional Drive upload volume. Adjust the total optimizer-step target to the complete target number, not an additional number of steps. On resume, keep the data, tokenizer, batch size, sequence length, gradient accumulation and learning-rate contract identical. Increasing the target deliberately extends the schedule.

## Checkpoint and interruption protocol

Every saved checkpoint generation contains:
- the full PyTorch training checkpoint, including model, optimizer, scheduler horizon and RNG state;
- the matching metrics log snapshot;
- the matching run manifest and data/config resume contract;
- SHA-256 hashes and file sizes in a generation manifest.

Publishing is ordered: upload each file to a temporary name, fsync it, verify its size/hash, rename it to its immutable generation name, and check the PyTorch ZIP/CRC. Only after all generation files validate does the runner atomically replace the single visible `training_progress.json` record. That one file is both the commit marker and the current progress box; it shows one current number, for example `current_step: 1250`, and the previous displayed number is replaced. A generation uploaded before that atomic write is treated as uncommitted and cleaned before the next save.

The last two complete checkpoint generations are kept for rollback. A partial upload is ignored. If the latest generation is damaged, restore falls back to the previous verified generation. A session lease/heartbeat prevents two active Colab notebooks from writing to the same checkpoint store; an abandoned lease is recoverable after three minutes without a heartbeat.

A runtime loss between save intervals can lose the work since the last verified checkpoint. The default interval limits this window to at most 9 completed steps, subject to interruption timing. An unavailable Drive mount stops the save path rather than advancing the progress number without a verified copy.

The progress state values include `not_started`, `checkpointed`, and `completed`. `completed` is written only after the final trained Q2-v2 export has also been copied and hash-verified in Drive. The initial model remains unchanged; trained deployment exports are saved separately under `trained_models/`.

## Important engineering boundaries

- The architecture remains attention-free: causal convolution + gated diagonal state-space + sparse Top-1 MoE. No Transformer attention or Q/K/V attention is added.
- No external pretrained model weights are downloaded or loaded.
- Q2-v2 matrix symbols are 2-bit. Row scales are FP32, one-dimensional parameters are FP16, and current QAT master parameters and AdamW optimizer state are FP32. This is not an all-state 2-bit trainer.
- The current single-device trainer is configured for the 0.7M Light model, subject to actual available memory. Distributed sharded training is not implemented and is outside this workflow.
- A saved checkpoint proves resumable training state, not assistant quality. Language capability still requires real licensed data, training, and independent held-out evaluations.


## Additional resume-safety invariants

- A Drive checkpoint writer must own the active training-session lease. Publishing without the lease fails closed.
- The visible progress step is monotonic. A stale writer may not publish a lower step over a newer committed checkpoint.
- Re-publishing an already committed step is accepted only if its checkpoint, metrics and run-manifest hashes match the committed generation exactly.
- The training runner does not restore or mutate checkpoint state in its display/preflight cell. It acquires the writer lease first, then restores the latest complete Drive generation while holding that lease.
- Resume compares data, tokenizer, config and optimizer settings while preserving the original step-zero model fingerprint. The original initialization lineage remains in later run manifests and checkpoints.
- A Python exception marks the progress status as `interrupted` where Drive is reachable. A hard Colab runtime loss is detected on the next launch when the prior status remains `running` but no live lease can be acquired.

## Private repository access from Colab

The Training-Ready Builder clones this private repository. Give Colab access to GitHub and either provide a fine-grained token in the Colab secret QMINI_GITHUB_TOKEN or enter it at the hidden prompt. Restrict the token to this repository with Contents: Read-only permission. It is passed only to the Git process as a temporary authentication header and is not written into the notebook, Drive package or repository.
