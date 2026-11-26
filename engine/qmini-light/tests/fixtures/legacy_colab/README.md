# Legacy Colab notebook fixtures

These six notebooks are preserved only because CI regression tests and the
legacy notebook validation workflow inspect their source structure.

They are **not active training entry points** and should not be launched as
current instructions. Use the supported notebooks indexed in
`colab/README.md`.

The fixtures contain no saved Colab outputs, private checkpoints or dataset
payloads. Keep fixture changes intentional and update their regression tests
and `.github/workflows/colab-initial-build.yml` together.
