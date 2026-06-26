# Release notes

Generated locally for manuscript deposition.

## Sanitization decisions

- No Materials Project API keys are included.
- No PET/MatterSim/MACE/CHGNet/SevenNet model checkpoints are included.
- No full Materials Project CIF cache is included.
- Raw text-mined synthesis corpus files are not included; only processed DOI-backed route-validation tables are included.
- Code paths were cleaned to use relative paths or environment variables where practical.

## Known limitations

This release is a processed-data and workflow-code package. It is sufficient for checking manuscript numbers and inspecting route-level records. Full end-to-end recomputation requires external credentials, database access and local model checkpoints.
