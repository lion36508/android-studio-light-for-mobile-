# Dataset source provenance and release gates

Training data must have a recorded source, revision, licence, language, collection date, transformation version and SHA-256 hash. Never guess a licence. Retain raw source files in a cache and never overwrite an existing verified dataset silently.

The legacy and Add/Sub implementations have separate data pipelines and manifests. Do not share prepared data or token streams between them.

For the legacy Q-Mini Light tokenizer and preparation flow, use the active configuration `configs/qmini_light_0.7m.json` and the tokenizer size that the selected dataset/build actually supports. The Light profile targets 4,096 vocabulary entries; do not use the retired 8,192-token/152M profile.

For Add/Sub, use its own manifest-bound importer/preparation scripts. The optional pinned preparation uses a fixed Simple English Wikipedia revision, filtered OASST1 and Dolly sources. It requires explicit licence acceptance and records split/file hashes. Local user-provided datasets must include their declared source and licence; these declarations are recorded but are not independent proof of rights or authorship.

## Release checks

- Source revision and licence are known and reviewable.
- Exact source and converted-split hashes are recorded.
- Train/validation/test splits are separate and reproducible.
- Tokenizer is fitted using training data only.
- Duplicate documents/conversations do not cross splits.
- Checkpoint metadata binds model configuration, tokenizer and dataset hashes.
- Held-out evaluation data is never used for fitting or training.

The current filters are a first engineering gate, not a complete privacy, legal or language-identification audit. Manual source review remains necessary.
