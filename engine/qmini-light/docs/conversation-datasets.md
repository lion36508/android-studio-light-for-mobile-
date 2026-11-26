# Conversation datasets for Q-mini-light-0.7M

Training no longer uses TinyStories. The 0.7M model trains on real human
conversation, English only, from two datasets:

- **DailyDialog** — human-written two-speaker daily English dialogues.
- **OpenAssistant oasst1** — human-generated, human-annotated assistant-style
  conversation trees, filtered to English only.

The model architecture and the 0.7M config (`configs/qmini_light_0.7m.json`)
are unchanged. The existing tokenizer (vocab 4096) is reused, not retrained.

## Provenance (verified on Hugging Face, 2026-11-17)

### DailyDialog — `roskoN/dailydialog`, config `full`

- Repository revision: `5214b2a66405abf87fd229e5c1007985501ffe3e`
- License: **CC BY-NC-SA 4.0** (non-commercial — see the note below)
- Splits: train 11,118, validation 1,000, test 1,000 (13,118 total)
- Columns: `id` (string), `acts` (Sequence int8), `emotions` (Sequence int8),
  `utterances` (Sequence string)
- Source files (verified by SHA-256 and size by the preparation script):

  | file | bytes | SHA-256 |
  | --- | --- | --- |
  | `train.zip` | 1,935,190 | `c5179ab5a9a86a77b9d29114087c6824c4b364abea2a53a0c24be761133f7` |
  | `validation.zip` | 180,450 | `1dbcb00a0b4fe709788ce6604b561ba76557c88fe41f049d7899287838f0ec6a` |
  | `test.zip` | 178,601 | `cfffc0bab93a3697c2562a9e35c4533f0de2a93e86a4e4d3ea45cbde4bea00f` |

- About 7.9 turns per dialogue, about 102,979 utterances.

### OpenAssistant oasst1 — `OpenAssistant/oasst1`, English only, train split

- Repository revision: `fdf72ae0827c1cda404ff25b6603abec9e3399b`
- License: **Apache-2.0**
- Splits: train 84,437 rows, validation 4,401 (88,838 total)
- Train parquet file: `data/train-00000-of-00001-b42a775f407cee45.parquet`,
  39,516,251 bytes, LFS SHA-256
  `bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e`
- Columns include `message_id`, `parent_id`, `text`, `role`
  (`prompter`/`assistant`), `lang`, `deleted`, `message_tree_id`; rows are in
  depth-first order.
- The English message count is counted at runtime by the preparation script
  and logged; it is not hard-coded. (The README's "English: 71,956" is the
  total English count across the whole dataset, not the train split alone.)

## License note

DailyDialog is **CC BY-NC-SA 4.0 — non-commercial**. The conversation run
therefore inherits a non-commercial restriction from that dataset. If the
model or its outputs are ever used commercially, DailyDialog must be removed
from the training data first. OASST1 alone is Apache-2.0.

## Prepared data (Google Drive)

The preparation script is run by the Colab notebook (V11); it writes:

- `MyDrive/QMini-Light-0.7M/datasets/dailydialog/dailydialog_train.jsonl`
- `MyDrive/QMini-Light-0.7M/datasets/dailydialog/dailydialog_validation.jsonl`
- `MyDrive/QMini-Light-0.7M/datasets/dailydialog/dailydialog_test.jsonl`
- `MyDrive/QMini-Light-0.7M/datasets/dailydialog/manifest.json`
- `MyDrive/QMini-Light-0.7M/datasets/oasst1_en/oasst1_en_train.jsonl`
- `MyDrive/QMini-Light-0.7M/datasets/oasst1_en/manifest.json`

Each JSONL record is `{"messages": [{"role": "user" | "assistant", "content": ...}]}`.
DailyDialog turn 0 is the user, turn 1 the assistant, alternating. OASST1
`prompter` maps to `user` and `assistant` to `assistant`; each message tree
becomes exactly one dialogue (first-child chain, depth-first), so a tree can
never be split across the pipeline's train/validation split. Non-English,
deleted and empty messages are dropped; dialogues with no assistant turn are
dropped. Each `manifest.json` records the source repository, revision,
license, verified file hashes, and exact dialogue/utterance counts.

No TinyStories data is deleted by this change. The previous prepared
TinyStories token streams stay inside the training package as
`dataset-previous-v1/` when the notebook switches the package dataset.

## How training runs on the conversation data

1. Open the CPU resume notebook `colab/QMini_English_CPU_One_Click_Resume.ipynb`
   (V11) in Colab with the owner's Drive mounted.
2. The notebook runs `scripts/prepare_conversation_datasets.py` from the
   fetched repository source, which downloads, verifies and converts both
   datasets into Drive (skipping them when the prepared files already verify).
3. The notebook verifies the prepared JSONL files against their manifests,
   then prepares token streams with the existing tokenizer at
   `valid_fraction=0.02` and `--loss-mask assistant` into the existing
   training package `QMini-English-512-Q2v2-e1a187f0ca1567cd`.
4. Training resumes from `training_state/checkpoint-step-000000012540.pt`
   (step 12,540/20,000) with `--allow-dataset-change`, which records the
   dataset switch (previous data fingerprints and switch step) in the new
   checkpoint's resume contract. It never restarts from random.

## Why the resume needs an explicit switch

`qmini/train.py` records the training data (token-stream and mask SHA-256,
tokenizer SHA-256, dataset-manifest SHA-256, batch/sequence length,
learning rate, weight decay, seed) in the checkpoint's resume contract, and
refuses to resume when it changes. A dataset change therefore cannot resume
silently. The new `--allow-dataset-change` flag (default off) is the only
way to continue on new data: it keeps the config, seed, learning rate,
weight decay and effective-batch checks, re-baselines only the data
fingerprints, and records the previous fingerprints and the switch step in
the checkpoint contract, so the switch is auditable in every later
checkpoint.

## Why the notebook prepares without `--source-metadata`

`qmini/data.py:prepare_token_streams` refuses to prepare data whose source
provenance differs from the tokenizer's sidecar manifest when
`--source-metadata` is given. The existing tokenizer was fitted on
TinyStories, so its sidecar pins TinyStories provenance. The notebook
therefore prepares the conversation streams without `--source-metadata`;
the conversation provenance lives in the Drive `manifest.json` files (and
in this document), while `dataset.json` still lists the four JSONL source
files with their SHA-256s.

## Running the preparation script by hand

```bash
python scripts/prepare_conversation_datasets.py \
  --datasets-dir /content/drive/MyDrive/QMini-Light-0.7M/datasets
```

Options: `--cache-dir` (download cache), `--force` (re-download and
re-convert), `--only dailydialog|oasst1` (prepare one dataset).
