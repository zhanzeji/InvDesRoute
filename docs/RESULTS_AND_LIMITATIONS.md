# Results and interpretation

| Evaluation | v0.2.1 result | Scope |
|---|---|---|
| Retrospective144, final top five | 62 strict / 91 strong-or-complete / 114 moderate+ | Fixed candidate pools; 86 records use literature-seeded species; 57 reference DOIs overlap the historical prior |
| Final top five without evaluation-template bonus | 80 strong-or-complete / 105 moderate+ | Same displayed routes; ranking template terms remain unchanged |
| Base-score AUC | 0.831504 | Same 5,326 candidates and 63 strict full-route positives for all predictors |
| Positive-control80, final top five | 78 complete / 80 moderate+ | Curated availability control, not a blind test |
| Temporal holdout | 47 strict / 49 strong+ / 53 moderate+ of 56 | Literature knowledge and ranking labels frozen; complete MLIP training and MP pool not time-reconstructed |
| External precursor endpoint | 64 strict / 79 strong+ / 110 moderate+ of 120 | 24 target clusters, five records each; model training exposure is not uniformly known |

The 107 hydride phases are reported by a collaborator to have 100% MPtrj and approximately 72% sAlex structural overlap. The per-phase matching table, tolerances, exact sAlex count and actual training-partition membership have not been supplied in the present workspace. These percentages are marked reported, not independently verified. The existing approximately 4 meV/atom phase error and 0.053 eV/written-reaction error are retained as reconstruction results on potentially training-covered references. They are not evidence of independent unseen-material generalization. Constructing new reaction combinations from training-covered phase energies does not remove phase-level overlap.

DFT calculation settings are not inferred merely from MP IDs or the MPtrj association. Matched structure/task metadata are required. No independent replacement energy test set was created in this release. Recommendations are hypotheses, and ranking scores/confidence labels are not calibrated experimental success probabilities.

Raw MLIP, MP formation-energy and aligned model/reservoir conventions are different. Aggregate differences across these conventions are not pure MLIP errors. The fixed 0.05 eV/target margin is operational, not a calibrated 2-sigma rule. Gas-only corrections omit solid vibrational contributions.

The computational scope of this archive is defined by DATA_INDEX.csv and the reproduction commands. Additional compatibility-correction, gas-reference reconstruction or relaxation results require the corresponding calculation inputs and outputs for independent reproduction. Manuscript sources are supplied separately from the software and numerical-data ZIPs.
