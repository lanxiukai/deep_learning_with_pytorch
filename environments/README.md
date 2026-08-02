# Conda Runtime Record

The `d2l` directory stores an exact snapshot of the tested training and lesson
environment. The root `environment.yml` remains the human-maintained,
declarative dependency source; the generated record adds exact Conda artifacts,
the complete pip package state, CUDA-index routing, checksums, and repository
revision.

The local `dl-utils` package is intentionally excluded from the pip lock and is
restored from this repository with an editable install. This keeps the record
portable and avoids machine-specific absolute paths.

The unified manager and documentation live in the sibling
`ai-agent-framework` repository at `config/conda/`. Do not edit generated lock
files manually; refresh them after an intentional environment update and
successful project verification.
