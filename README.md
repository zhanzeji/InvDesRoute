# InvDesRoute release package

This release package accompanies the manuscript **"InvDesRoute: route-network planning for inorganic synthesis"**.
It is organized as a public data-and-code deposit for journal submission, revision, or reviewer inspection.

## Directory structure

- `data/`: processed benchmark tables, literature-route validation outputs, score/risk diagnostics and figure source data.
- `code/`: cleaned workflow code for phase retrieval, PET-OAM-XL energy caching, reaction enumeration, route scoring and literature-route comparison.
- `manuscript/`: manuscript and supplementary LaTeX snapshots for version context only.
- `docs/`: release notes, data/code availability text, upload checklist, data dictionary and reproducibility notes.
- `MANIFEST.txt`: full file list.
- `SHA256SUMS.txt`: SHA256 checksums for integrity checks.

## Main data included

The package contains processed data generated in this study, including:

1. model-evidence benchmarks for PET-OAM-XL and baseline interatomic potentials;
2. the 120-target inorganic route-network benchmark and 17,085 PET/MP-form comparable reaction records;
3. DOI-backed literature-route validation outputs for the initial validation set and combined 145-route audit;
4. InvDesRoute scoring, diversity-aware ranking, precursor-class diagnostics and refinement queues;
5. Li2AgH6/Li2AuH6 pressure-stress-test summary tables;
6. source tables used to create the main and supplementary figures.

## Main code included

The code snapshot contains scripts and modules for:

- Materials Project phase retrieval and local cache handling;
- PET-OAM-XL phase-energy evaluation through local uPET/PET configuration;
- balanced formation/decomposition reaction enumeration;
- MP-form reference construction from Materials Project formation energies;
- practical-route filtering, InvDesRoute scoring and diversity-aware display;
- DOI-backed literature-route comparison and route-family similarity analysis;
- precursor-class diagnostics and gas-reservoir sensitivity analysis.

## External resources not redistributed

- Materials Project raw database content and full CIF cache are not redistributed. Users should query Materials Project under its access terms for raw structures.
- Materials Project API keys are not included.
- PET-OAM-XL, MatterSim, MACE, CHGNet and SevenNet model checkpoints are not redistributed. Users should obtain model files from their original sources and configure paths locally.
- The original public text-mined synthesis corpus is not claimed as a new dataset here. Processed route-validation tables derived from it are included; users should cite and access the original Kononova/Ceder resource when rerunning extraction from scratch.

## Suggested upload workflow

1. Push this folder to a public GitHub repository for code browsing and issue tracking.
2. Upload `InvDesRoute_release_20260626.tar.gz` to Zenodo/Figshare/OSF for a persistent data DOI.
3. If using Zenodo-GitHub integration, archive the GitHub release to obtain a code DOI.
4. Replace `ZENODO_DOI_OR_URL`, `GITHUB_OR_ARCHIVE_URL` and optionally `CODE_ARCHIVE_DOI_OR_URL` in `docs/DATA_AND_CODE_AVAILABILITY_TEXT.md` after upload.
5. Update the manuscript Data availability and Code availability sections with the final links.
6. Keep `MANIFEST.txt` and `SHA256SUMS.txt` with the uploaded package.

## Integrity check

After download, verify the package with:

```bash
cd InvDesRoute_release
sha256sum -c SHA256SUMS.txt
```

## Notes

See:

- `docs/DATA_DICTIONARY_CN.md` for a Chinese explanation of each data group.
- `docs/UPLOAD_CHECKLIST_CN.md` for upload and manuscript wording instructions.
- `docs/REPRODUCIBILITY_NOTES_CN.md` for rerun requirements and limitations.
- `docs/DATA_AND_CODE_AVAILABILITY_TEXT.md` for manuscript-ready availability text after replacing placeholders.
- `GITHUB_UPLOAD_COMMANDS.md` for the local Git commands to push this prepared repository.
