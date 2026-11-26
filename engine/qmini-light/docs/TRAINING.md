# Training and data protocol

This repository has two independent training paths. Use the matching package and Colab notebook; never mix their datasets, tokenizers, checkpoints or Drive directories.

## Legacy Q-Mini Light

- Package: `qmini/`.
- Configuration: `configs/qmini_light_0.7m.json`.
- Supported Colab: `colab/QMini_English_CPU_One_Click_Resume.ipynb`.
- Existing-checkpoint chat: `colab/QMini_Existing_Drive_Data_CPU_Chat.ipynb`.

Preserve the legacy checkpoint format and existing Drive artifacts. Review the notebook and its dataset manifests before starting or resuming a run.

## Q-Mini Add/Sub

- Package: `qmini_addsub/`.
- Supported Colab: `colab/QMini_AddSub_V1_From_Scratch.ipynb`.
- Uses its own tokenizer, dataset manifest, pretraining/SFT directories and checkpoint formats.
- Data preparation, tokenizer construction, pretraining, SFT and evaluation are disabled by default in the notebook.

The optional pinned data path uses Simple English Wikipedia, filtered OASST1 and Dolly sources. Review each licence before enabling source downloads. A local user-provided dataset can be imported with explicit source/license declarations. The importer records hashes and split metadata but cannot independently prove the truth of the declarations.

## Reproducibility requirements

Every run should retain the model configuration, exact tokenizer, train/validation/test hashes, source/licence metadata, seed, optimizer settings, step count and checkpoint lineage. Resume only from a checkpoint whose complete run contract matches. Do not silently reuse artifacts from another model or dataset.

## CI boundary

CI runs unit/regression tests, synthetic learning proofs, small smoke training, package/native build checks and lexical evaluation fixtures. It does not run the full English pretraining or SFT recipe. A green CI run confirms tested software paths only; it does not establish real-world assistant quality.

The trainers are single-process and single-device. Distributed training and production Android deployment are not delivered by the current code.
