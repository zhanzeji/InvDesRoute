# Reproducing numerical results

The commands below require the original analysis inputs. The public CC BY 4.0 ZIP provides benchmark summaries and validation flags for inspection. See [Data sources](DATA_SOURCES.md) for the upstream references.

## Baseline analyses

`python code/reproduce.py --output recomputed/basic` reconstructs reaction differences from phase energies and coefficients, computes the core120 metrics and re-evaluates the normalized precursor endpoint. Inspect `checks.json`, `summary.json`, `seven_model_recomputed.csv`, `core120_by_target.csv` and `external_common_endpoint_summary.csv`.

Expected baseline checks include 120 targets, 18,063 reactions, core top-1 agreement 100/120, mean sign 0.9316879531589801 and mean top-five overlap 3.691666666666667. The Equiformer reaction MAE is 0.05311591011558839 eV/written reaction. Checks verify agreement with frozen expected values, not independent certification of their reference conventions.

`--full-analysis` also recomputes development policy selection, later-target evaluation, literature ranking ablations and transfer to the external benchmark. It may take several minutes on a CPU. All nine built-in checks should pass. Logs should be retained alongside newly computed outputs.

## New inference

Use the archived upstream source in `code/vendor/equiformer_v3`, the observed package-version record and an appropriate CUDA/PyTorch environment. A generic newer Fair-Chem installation is not necessarily compatible with this architecture. The exact checkpoint is required; `code/replay_gpu.py` verifies its hash.

```sh
python code/replay_gpu.py --checkpoint /path/to/omat24-mptrj-salex_gradient.pt --cohort hydride --output recomputed/fresh_hydride.csv
```

The output contains fresh phase energies and metadata. It is not a rerun of all original DFT or language-model inference. The analysis requirements are not a GPU-environment lockfile. Follow upstream dependency/build instructions and validate a single structure first.

## Additional diagnostics

Run `scripts/setup_revision.py` after data installation. The numerical tools under `revision/tools` use the same frozen phase energies and candidates. `continue_MP_compatibility.py` requires authentic MP entries and a personal API key supplied through the environment. It has not completed an authenticated full-cohort corrected benchmark. Do not fill missing Hubbard or POTCAR values with fabricated defaults.

The archive includes historical source paths for traceability. Supported baseline reproduction uses package-relative paths. Some original retrieval/application scripts require local path migration; their presence is not a promise of a turnkey live screening service.
