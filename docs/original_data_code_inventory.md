# Data and code availability inventory

This note lists the concrete assets that support the manuscript `final_paper/main_text/nc.tex` and are intended for the corresponding Data availability and Code availability statements.

## Data package

### 1. Main 120-target benchmark
Directory: `final_paper/support_tables/synthesis_pathway_screening_mp_pet_benchmark_120/`

Key files:
- `mp_pet_benchmark_summary.csv`
- `mp_pet_benchmark_category_summary.csv`
- `mp_pet_benchmark_summary_with_categories.csv`
- `mp_pet_reaction_data.csv`
- `synthesis_recommendation_summary_all.csv`
- `recommended_routes_top5_all.csv`
- `recommendation_category_summary.csv`
- `top1_route_feature_summary.csv`
- `dft_refinement_queue.csv`
- `derived_pathway_analysis.csv`
- `derived_pooled_reaction_stats.csv`
- `README_summary.md`
- `DERIVED_ANALYSIS.md`
- `EXTENDED_PAPER_ANALYSIS.md`
- `RECOMMENDATION_LAYER_ANALYSIS.md`

Use in manuscript:
- PET-vs-MP benchmark statistics
- target/category summaries
- route rankings and practical-route filters
- DFT/manual refinement queue
- route-complexity analysis

### 2. Model-evidence package
Directory: `final_paper/support_tables/model_evidence/`

Key files:
- `compound_model_metrics.csv`
- `reaction_model_metrics.csv`
- `batch_target_summary.csv`
- `batch_top5_dft_paths.csv`
- `batch_model_summary_aggregated.csv`

Use in manuscript:
- PET-OAM-XL model selection
- compound-level and reaction-level accuracy comparisons
- batch DFT/pathway validation

### 3. Literature-route benchmark package
Directory: `final_paper/support_tables/literature_route_benchmark/`

Key files:
- `literature_routes_seed.csv`
- `literature_routes_seed_updated.csv`
- `targeted_final_seed_literature_route_eval.csv`
- `route_similarity_eval.csv`
- `route_similarity_summary.csv`
- `literature_route_match_summary.csv`
- `literature_route_match_top5.csv`
- `literature_route_match_category_summary.csv`
- `literature_route_match_precursor_scope_summary.csv`
- `literature_route_family_match.csv`
- `top_synthesizability_recommendations.csv`
- `synthesizability_summary_all.csv`
- `synthesizability_routes_top10_all.csv`
- `synthesizability_diverse_top10_all.csv`
- `synthesizability_diverse_top20_all.csv`
- `recommendation_ablation_summary.csv`
- `recommendation_ablation_detail.csv`
- `diverse_recommendation_summary.csv`
- `synthesizability_refinement_queue.csv`
- `route_similarity_refinement_queue.csv`
- `route_similarity_refinement_precursor_counts.csv`
- `top_synthesizability_risk_flags.csv`
- `top_synthesizability_template_summary.csv`
- `condition_sensitivity_practical_summary.csv`
- `synthesizability_literature_summary.csv`
- `literature_synth_rank_eval.csv`
- `SYNTHESIZABILITY_AGGREGATE_REPORT.md`
- `SYNTHESIZABILITY_CLAIM_SUPPORT_AUDIT.md`
- `SYNTHESIZABILITY_RECOMMENDATION_ANALYSIS.md`
- `RECOMMENDATION_ABLATION_REPORT.md`
- `ROUTE_SIMILARITY_VALIDATION.md`
- `ROUTE_SIMILARITY_REFINEMENT_QUEUE.md`
- `CONDITION_SENSITIVITY_ANALYSIS.md`
- `MANUAL_PRECURSOR_THERMOCHEMISTRY.md`
- `NEXT_STEP_ROUTE_FAMILY_AND_DIVERSITY_REPORT.md`
- `SEED80_FINAL_VALIDATION_REPORT.md`
- `download_ceder_text_mined_dataset.py`
- `extract_ceder_text_mined_routes.py`

Use in manuscript:
- 80-target positive control
- 59-route DOI-backed overlap audit
- 145-route combined literature audit
- route-family similarity definitions
- synthesizability-score ablation
- diversity-aware recommendation analysis
- condition-sensitivity summary
- refinement queues and risk flags

### 4. Literature-seeded extension package
Directory: `final_paper/support_tables/literature_seeded_extension_cached_53/`

Key files:
- `combined_benchmark_summary.csv`
- `combined_literature_similarity_summary.csv`
- `combined_literature_precursor_class_diagnostics.csv`
- `combined_literature_rank_bucket_diagnostics.csv`
- `combined_literature_target_complexity_diagnostics.csv`
- `combined_literature_top5_miss_refinement_queue.csv`
- `combined_recommendation_ablation_summary.csv`
- `combined_diverse_recommendation_summary.csv`
- `combined_literature_diagnostic_summary.csv`
- `selected_literature_routes.csv`
- `mp_pet_benchmark_summary.csv`
- `mp_pet_reaction_data.csv`
- `synthesizability_summary_all.csv`
- `synthesizability_routes_top10_all.csv`
- `synthesizability_diverse_top10_all.csv`
- `synthesizability_diverse_top20_all.csv`
- `route_similarity_eval.csv`
- `route_similarity_summary.csv`
- `recommendation_ablation_summary.csv`
- `diverse_recommendation_summary.csv`
- `CACHED_LITERATURE_EXTENSION_REPORT.md`
- `COMBINED_LITERATURE_FAILURE_MODE_DIAGNOSTICS.md`

Use in manuscript:
- 53 non-overlapping literature-seeded targets
- 86 additional DOI-backed route records
- combined 173-target audit
- combined 145-route validation
- precursor-class diagnostics
- combined recommendation ablation

## Code package

Directory: `final_paper/code_snapshot/`

Key files:
- `run_screening.py`
- `benchmark_mp_pet.py`
- `aggregate_recommendations.py`
- `plot_extended_paper_analysis.py`
- `plot_recommendation_layer.py`
- `src/analysis.py`
- `src/common.py`
- `src/pet_energy.py`
- `src/phase_sources.py`
- `src/reactions.py`
- `examples/KI_mp.sh`
- `examples/NaAlH4_local.sh`
- `docs/BENCHMARK_ANALYSIS.md`
- `docs/PROJECT_STRUCTURE.md`
- `README.md`

Use in manuscript:
- formula/CIF target handling
- MP phase retrieval
- PET-OAM-XL energy evaluation through the local uPET/PET environment
- balanced reaction enumeration
- route ranking and recommendation aggregation
- benchmark generation
- literature comparison and diversity-aware display
- plotting and paper figure generation

## Manuscript-facing release package

For submission, the public archive should include:
- the processed CSV/MD source-data tables listed above
- the frozen configuration files used for the main runs
- the environment specification for the local PET/uPET and Materials Project workflow
- the scripts required to regenerate the benchmark tables and manuscript figures
- the example input files for formula and CIF runs
- a short reproducibility manifest mapping each manuscript figure/table to the source data file(s)

