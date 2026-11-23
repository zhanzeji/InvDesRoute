# Data layout and field conventions

All text tables use UTF-8 CSV unless a different extension is shown. Empty cells and NaN denote unavailable values, not zero. Phase formulas are reduced using the archived pymatgen convention; phase identity also depends on structure and source. Composition identity alone does not distinguish polymorphs.

## Structure provenance

`data/structure_manifest.csv` maps `original_path` to a package-relative `archive_path`, records the structure SHA256 and lists cohort membership. `data/structures/` contains the corresponding CIFs. Use the archive path on a new machine. Do not rewrite the original-path column and then call it original provenance. A structure content hash identifies the input bytes, not a database release or a DFT calculation protocol.

## Hydride reference and prediction tables

`data/hydride/input/mp_dft_phase_energies.csv` contains the archived formula/MP identifier and reference energies. FAILED or non-numeric values are excluded from numerical comparison. The reference table does not contain complete run-type, Hubbard, PAW and original calculation settings for every entry.

`data/hydride/model_predictions/*_phase_energies.csv` contains model phase outputs. Relevant fields include `formula`, `energy_cell_eV`, `z`, `energy_per_formula_eV`, `status` and `error`. Here z is the number of reduced formula units in the cell. Reproduction retains successful predictions and uses the same formula deduplication rule as the archived benchmark.

`data/hydride/input/reactions_416.csv` supplies the balanced written expressions. Reaction energy is the stoichiometric product sum minus reactant sum. Hydride MAE and RMSE are evaluated per written reaction, so multiplying the written expression also multiplies its energy/error.

## Core120 target directories

Each directory under `data/core120/targets/` describes one target. `phase_pool_raw.csv` stores retrieved phase metadata; `phase_pool_equiformer_v3.csv` adds model phase energies. `phase_stability.csv` contains the MP reference and phase descriptors used by the analysis. `production_form_aligned_phase_energy_sources.csv` records whether each recommendation energy is model-aligned, a prescribed gas/reservoir value, fallback or missing.

`formation_reactions.csv` and `decomposition_reactions.csv` contain `reaction`, `reactants`, `products`, `target_coefficient` and energy columns. Structured sides use `coefficient*formula | coefficient*formula`. Important columns are:

| Column | Meaning |
| --- | --- |
| `DeltaE_MP_FORM_per_target` | Reaction difference constructed from archived MP formation energies, divided by target coefficient |
| `DeltaE_EquiformerV3+DeNS-OAM2_per_target` | Raw all-structure model reaction difference per target |
| `DeltaE_PRODUCTION_FORM_ALIGNED_per_target` | Recommendation convention combining aligned model phases and explicitly prescribed reservoirs |
| `PRODUCTION_FORM_ALIGNED_missing_species` | Species preventing a complete aligned energy |
| `practical_route` | Heuristic phase/coefficient filter, not an experimental annotation |
| `energy_score` and other component scores | Inputs to the archived ranking policy |
| `synth_rank_deltaE_per_target` | Selected energy convention used in a particular scoring record |

Historical PET-named columns remain in some tables for traceability. Their presence does not make them current production energies. Avoid selecting a column by its position; select its full name and check its source convention.

## Literature and temporal cohorts

`data/literature/references/` contains normalized literature records. The 59-record and 86-record references form the 145-record retrospective evaluation. Literature augmentation differs by cohort and is not hidden. The seed80 positive control is a separate collection.

`data/temporal/split/` contains historical, development and later-target records. Important fields include `route_id`, `target`, `doi`, `publication_year`, `literature_precursors`, `literature_byproducts`, `source_index` and `literature_route_text`. A source index refers to a mined record, not a DFT task. First appearance means first appearance in the dated corpus, not a verified global discovery date.

`data/temporal/cases/` contains candidate networks. `data/temporal/archived_results/` records frozen protocol, development selection, displayed recommendations and source/DOI manifests. Additional November diagnostic outputs are in `revision_results/` in the data archive. They must not be substituted into an older figure while leaving its cohort and policy label unchanged.

## Released-model comparison

`data/external/benchmark_24x5.csv` contains 120 literature records for 24 targets. Prediction records preserve the model, target, rank and precursor output. The supported evaluator normalizes formulas and requires complete target-element coverage. Invalid candidates do not receive replacement generations. Targets without valid outputs remain failures in the denominator.

Strict precursor-set identity is separate from family similarity. Strong and moderate counts are cumulative. A target has five reference records but only one displayed list of up to five candidates. It is not prompted separately with each reference label.

## Additional evidence and applications

`audit_data/` contains numerical source tables required by the additional-analysis tools, including pressure phase energies and candidate discrimination labels. It is not an internal editorial review log. `applications/Li2AgH6/` supplies the later Ag-hydride case tables and structures. The Au-hydride source is under `data/Li2AuH6/`.

The pressure records distinguish cell E, V, PV and H. H already includes PV. Finite-pool model hulls and source-dependent molecular-H2 references are marked separately; they are not full DFT phase diagrams or pressure-relaxed hydrogen standards.
