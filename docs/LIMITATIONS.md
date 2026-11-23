# Scope and unresolved evidence

1. Core120 compares raw MLIP reaction differences with archived MP formation-energy differences. Elemental reference shifts cancel in balanced reactions; phase-dependent compatibility adjustments need not. MP2020 corrections have not been applied to all archived model entries in a matched comparison.
2. The MP release and authenticated retrieval time were not saved. File timestamps support approximate core cache construction on 29 April 2026; a version inferred from a release calendar is not a recorded version identifier.
3. DFT reference tables are available, but complete per-phase original settings/output correspondence and overlap with all OMat24/MPtrj/sAlex training structures have not been certified.
4. The fixed −0.05 eV/target rule is operational. Pooled MAE/RMSE is not a calibrated per-route sigma. No experimental success probabilities are claimed.
5. The retrospective 145-record networks include literature precursor augmentation. They do not measure blind precursor generation. The temporal test freezes literature knowledge, not the whole structural database or MLIP pretraining.
6. JANAF gas-only sensitivity omits solid free energies and lacks a uniformly certified static reservoir/ZPE baseline. It did not improve common-subset energy AUC in the added diagnostic.
7. Pressure calculations are single points on supplied structures. A pressure-matched H2 reference at 10 GPa and complete DFT hulls are missing. Finite-pool model hull values do not certify equilibrium stability.
8. Released-model comparisons are retrospective. Holm correction over eight reported paired tests leaves no difference resolved at 0.05. Performance depends on endpoint and valid-output target subset.

These limitations are part of the release record and should accompany downstream reuse. A passing checksum or reproduction check does not remove them.
