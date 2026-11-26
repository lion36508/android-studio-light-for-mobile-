# Building a Q2-v2 initial brain

This process creates an **untrained, random-initialized Q-Mini model** whose deliverable is the repository's Q2-v2 packed artifact. It does not train, read datasets, create an optimizer, or create a dense full-model checkpoint.

## Google Colab

This is a CLI build-only workflow, not a currently supported Colab notebook. Clone the repository, install its dependencies, and run the command below with the active Light configuration. It creates an untrained random-initialized Q2-v2 artifact; it does not train the model.

Mount Drive and clone the repository:

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone https://github.com/lion36508/Q-mini-light-0.7M.git
%cd Q-Mini
!pip install -e .
```

Run the build into a new, unique folder. The builder refuses to overwrite an existing folder:

```bash
REVISION=$(git rev-parse HEAD)
qmini build-initial \
  --config configs/qmini_light_0.7m.json \
  --output-dir /content/drive/MyDrive/QMini-Light-0.7M/initial_brains/QMini-Light-0.7M-Q2v2-initial-001 \
  --source-revision "$REVISION"
```

Verify the artifact:

```bash
qmini verify-export \
  --path /content/drive/MyDrive/QMini-Light-0.7M/initial_brains/QMini-Light-0.7M-Q2v2-initial-001/model.qmini.npz
```

## Output

- config.json — exact model configuration and seed.
- model.qmini.npz — Q2-v2 two-bit matrix weights, FP32 row scales and FP16 one-dimensional parameters.
- model_manifest.json — parameter count, Q2-v2 contract, artifact size and SHA-256, source revision, and explicit zero training/optimizer steps.

The builder creates model components sequentially in the same order as QMiniMoE initialization. It packs each component before releasing it, verifies the Q2 archive, and uses the packed/meta loader to validate every expected parameter name and shape without allocating all dense 0.7M weights.

## Precision contract

Q2-v2 is the production/deployment matrix-weight format. During later Quantization-Aware Training (QAT), Q-Mini uses FP32 master parameters and optimizer state with a straight-through estimator. Those are trainable state, not the deployed matrix weights. Activations are separately fake-quantized to signed INT8. The Q2-v2 codebook must not be replaced by FP16/BF16, INT8, INT4 or ternary production weights.

## Training is a separate stage

The initial artifact is not trained and does not include a tokenizer. The later training workflow must provide matching Q-Mini data streams and tokenizer. The 0.7M Light profile is intended to use the repository's existing single-device CUDA trainer; FSDP/ZeRO is not a prerequisite for this small profile. Distributed training is not included in this Light workflow. QAT still uses FP32 master parameters, gradients and optimizer state, so a packed 2-bit deployment artifact does not mean training itself runs in 2-bit memory.

Do not delete older FP32 shards as part of this migration. Keep them as a separate backup/diagnostic artifact until the new Q2-v2 output has been independently verified.

## Private repository access from Colab

The repository is public. Do not place a GitHub token in the notebook or output logs.
