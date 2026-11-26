# Results and interpretation

| Evaluation | v0.2.0 result | Scope |
|---|---|---|
| Retrospective145, final top five | 62 strict / 92 strong-or-complete / 115 moderate+ | Fixed candidate pools; 86 records use literature-seeded species; 58 reference DOIs overlap the historical prior |
| Final top five without evaluation-template bonus | 81 strong-or-complete / 106 moderate+ | Same displayed routes; ranking template terms remain unchanged |
| Base-score AUC | 0.831504 | Same 5,326 candidates and 63 strict full-route positives for all predictors |
| Positive-control80, final top five | 78 complete / 80 moderate+ | Curated availability control, not a blind test |
| Temporal holdout | 47 strict / 49 strong+ / 53 moderate+ of 56 | Literature knowledge and ranking labels frozen; complete MLIP training and MP pool not time-reconstructed |
| External precursor endpoint | 64 strict / 79 strong+ / 110 moderate+ of 120 | 24 target clusters, five records each; model training exposure is not uniformly known |

The 107 hydride phases are reported by a collaborator to have 100% MPtrj and approximately 72% sAlex structural overlap. The per-phase matching table, tolerances, exact sAlex count and actual training-partition membership have not been supplied in the present workspace. These percentages are marked reported, not independently verified. The existing approximately 4 meV/atom phase error and 0.053 eV/written-reaction error are retained as reconstruction results on potentially training-covered references. They are not evidence of independent unseen-material generalization. Constructing new reaction combinations from training-covered phase energies does not remove phase-level overlap.

DFT calculation settings are not inferred merely from MP IDs or the MPtrj association. Matched structure/task metadata are required. No independent replacement energy test set was created in this release. Recommendations are hypotheses, and ranking scores/confidence labels are not calibrated experimental success probabilities.

Raw MLIP, MP formation-energy and aligned model/reservoir conventions are different. Aggregate differences across these conventions are not pure MLIP errors. The fixed 0.05 eV/target margin is operational, not a calibrated 2-sigma rule. Gas-only corrections omit solid vibrational contributions.

Unverified collaborator MP2020-correction, gas-reference reconstruction and relaxation calculations have not been added as certified numerical outputs. The revised supplement remains a draft with unresolved items and is retained outside the public software/data artifacts.
