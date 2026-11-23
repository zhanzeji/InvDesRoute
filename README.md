# InvDesRoute

InvDesRoute builds balanced inorganic reaction networks from target formulas or crystal structures. The workflow combines phase records, interatomic-potential energies, common precursor and gas-reservoir entries, interpretable ranking factors, and explicit refinement flags.

The repository supports two distinct activities: reproducing archived numerical analyses, and inspecting the source used for phase retrieval, reaction enumeration and recommendation. Reproducing archived numbers does not require an MP API key, model weights or a GPU. Fresh energy inference and live database queries have separate requirements.

## Scope of the release

The production checkpoint used in the archived calculations is EquiformerV3+DeNS-OAM2, file `omat24-mptrj-salex_gradient.pt`, SHA256 `429ccded98163122e7ba588d78e2441653f37f3e091e106c432807fe373c8f98`. Weights are obtained separately from their provider. The archived hydride benchmark has 107 usable phase references and 416 balanced reactions; the core route benchmark has 120 targets and 18,063 formation reactions. Literature evaluations, temporal tests and external-model comparisons use different cohorts and endpoints. See [benchmarks](docs/BENCHMARKS.md).

## Install and reproduce

The public data archive contains processed benchmark summaries and archived validation flags under CC BY 4.0. Extract the ZIP to inspect these results. Running the full workflow requires the original benchmark inputs as well as the source-specific prerequisites described in [data sources](docs/DATA_SOURCES.md) and [reproduction](docs/REPRODUCING.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `code/reproduce.py` | Recompute baseline benchmark metrics from archived predictions and coefficients |
| `code/replay_gpu.py` | Fresh inference on archived structures with the specified checkpoint |
| `code/workflow/` | Archived workflow implementation; some historical commands retain server paths |
| `code/vendor/` | Upstream model implementation with original license notices |
| `revision/tools/` | Additional null-space, gas, matching, pressure and statistical analyses |
| `scripts/` | Data installation, validation, additional-analysis setup and release helpers |
| `docs/` | Cohorts, units, source terms, known limitations and publication instructions |
| `release-config.json` | Author-reviewed release metadata; no invented URLs or DOIs |

## Additional analyses

```sh
python scripts/setup_revision.py
python revision/tools/compute_supplements.py nullspace
python revision/tools/compute_supplements.py matching
python revision/tools/compute_supplements.py pressure
python revision/tools/compute_supplements.py statistics
python revision/tools/gas_thermochemistry.py
```

These commands write `revision/results/`. They require the original analysis inputs rather than new DFT calculations or new language-model generations. The setup creates a local relative directory layout needed by the archived analyses. It does not alter energies or labels.

## Scientific interpretation

Raw model energies, MP formation-energy references and model/reservoir-aligned recommendation energies are separate conventions. The archived core comparison did not apply MP2020 compatibility corrections to MLIP predictions. Authentic per-entry metadata are required for a matched corrected comparison. A fixed −0.05 eV/target margin is an operational rule, not a calibrated uncertainty estimate. No calibrated experimental success probability is supplied.

Training-data overlap, complete original DFT provenance and pressure-matched hydrogen references remain unresolved in parts of the analysis. Gas-only thermal increments are not complete reaction Gibbs energies. The Li–Au–H and Li–Ag–H applications are computational hypotheses with competing-phase limitations. Read [known limitations](docs/LIMITATIONS.md) before interpreting a recommended route.

## Data, citation and license

Processed data are distributed separately to keep Git history small and preserve a frozen numerical snapshot. The Zenodo identifiers for this release are [software](https://doi.org/10.5281/zenodo.24453272) and [processed data](https://doi.org/10.5281/zenodo.24453293). `CITATION.cff` provides the software citation metadata. The manuscript is a separate research output; this repository does not assert that all manuscript calculations have been independently rerun.

The original InvDesRoute code is distributed under the [MIT License](LICENSE). See [LICENSE.md](LICENSE.md) for scope. Third-party licenses and data terms remain applicable. Do not commit API credentials, model weights or private handover records. For release instructions, start with [PUBLISHING](docs/PUBLISHING.md).
