# Data directory

This directory contains processed data tables for the InvDesRoute manuscript. All files are CSV, JSON, TSV or Markdown unless stated otherwise.

## Directory map

### `support_tables/model_evidence/`
Model-comparison data used to justify PET-OAM-XL as the production energy engine.

Key files:
- `compound_model_metrics.csv`: phase-energy metrics over 107 compounds.
- `reaction_model_metrics.csv`: reaction-energy metrics over 416 balanced hydride reactions.
- `batch_model_summary_aggregated.csv`: target-level hydride pathway metrics over 37 hydride targets.
- `batch_target_summary.csv`: per-target hydride benchmark summaries.
- `batch_top5_dft_paths.csv`: DFT top-path records used for route-ranking comparisons.

### `support_tables/benchmark_120/`
The main 120-target inorganic route-network benchmark.

Key files:
- `mp_pet_benchmark_summary.csv`: target-level PET/MP-form benchmark summary.
- `mp_pet_reaction_data.csv`: 17,085 PET/MP-form comparable formation reactions.
- `synthesis_recommendation_summary_all.csv`: confidence labels and top route records for all 120 targets.
- `recommended_routes_top5_all.csv`: top-5 route recommendations per target.
- `recommendation_category_summary.csv`: category-level confidence and recommendation statistics.
- `dft_refinement_queue.csv`: targets flagged for higher-fidelity or expert refinement.

### `support_tables/literature_validation_initial_80/`
Initial curated and DOI-backed literature-route validation for the 80-target validation set.

Key files:
- `literature_routes_seed.csv` and `literature_routes_seed_updated.csv`: curated route seeds.
- `route_similarity_eval.csv`: route-family similarity evaluation.
- `recommendation_ablation_summary.csv`: ranking ablation for PET-energy-only, practical-first, score and diversity-aware modes.
- `synthesizability_routes_top10_all.csv`: synthesis-aware recommendation outputs.

### `support_tables/literature_validation_combined_145/`
Combined DOI-backed literature-route audit: initial overlap plus literature-seeded extension.

Key files:
- `selected_literature_routes.csv`: DOI-backed literature records selected for the extension.
- `combined_literature_similarity_summary.csv`: exact/strong/moderate route-family recovery metrics.
- `combined_recommendation_ablation_summary.csv`: combined ranking-ablation results over 145 routes.
- `combined_literature_precursor_class_diagnostics.csv`: precursor-class failure-mode diagnostics.
- `combined_literature_top5_miss_refinement_queue.csv`: cases present in the full network but missing from the top-5 recommendation window.

### `support_tables/ranking_and_risk_diagnostics/`
Additional diagnostics for score interpretation, precursor limitations, gas-reservoir sensitivity and refinement queues.

### `support_tables/li_hydride_pressure_test/`
Out-of-distribution hydride pressure-stress-test tables for Li2AgH6/Li2AuH6 cases.

### `figure_source_data/`
Curated source tables for generating the main and optional extended figures.

## Notes on units

- Phase MAE values are usually in eV atom^-1.
- Reaction energies and reaction-energy MAEs are normalized per target formula unit unless a file explicitly states otherwise.
- `MP-form` denotes reaction energies reconstructed from Materials Project formation energies, not direct total-energy add/subtract references.
