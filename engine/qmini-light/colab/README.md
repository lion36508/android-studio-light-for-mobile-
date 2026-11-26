# Supported Colab entry points

The repository has two model implementations. Their Drive folders, tokenizers, datasets and checkpoints are separate.

## Legacy Q-Mini Light

- Training/resume: [Q-Mini English CPU resume](QMini_English_CPU_One_Click_Resume.ipynb)
- Existing-checkpoint chat: [Q-Mini existing Drive data chat](QMini_Existing_Drive_Data_CPU_Chat.ipynb)
- Package: `qmini/`
- Configuration: `configs/qmini_light_0.7m.json`

These notebooks work with the legacy checkpoint/data lineage only. Preserve existing Drive artifacts.

## Q-Mini Add/Sub

- From-scratch experimental pipeline: [Q-Mini Add/Sub](QMini_AddSub_V1_From_Scratch.ipynb)
- Package: `qmini_addsub/`
- Separate tokenizer, dataset manifest, pretraining/SFT checkpoints and evaluation.

Data preparation, tokenizer building, pretraining, SFT and evaluation are disabled by default. Review the selected source licence and data manifest before enabling them.

## Archive

Historical notebook fixtures under `tests/fixtures/legacy_colab/` are used only by regression tests. They are not active training instructions or additional models.
