# Benchmarks and evaluation endpoints

| Cohort | Size | Evaluation |
| --- | --- | --- |
| Complex hydrides | 107 phase references, 416 reactions, 37 targets | Phase energy, reaction difference and sign |
| Core network | 120 targets, 18,063 formation reactions | Raw-model/MP-form reaction agreement |
| Retrospective literature | 145 records, including 86 literature-augmented records | Candidate coverage and ranking visibility |
| Temporal development | 57 targets first recorded in 2021–2022 | Selection of prespecified score profiles and mixing/diversity parameters |
| Later corpus targets | 56 first recorded in 2023–2024 | Frozen literature-knowledge evaluation |
| Released-model comparison | 24 targets × 5 records, 120 records | Shared normalized precursor-set endpoint |

Hydride and pressure errors are eV per written reaction. Network energies are eV per target formula unit; phase errors are eV/atom. Core sign agreement is a target macro average. Top-five recovery means at least one of five candidates matches a reference record, not five independent correct predictions.

Historical complete-route recall allows extra species, whereas strict precursor-set identity does not. Strong and moderate thresholds are 0.78 and 0.62. Cumulative counts must not be stacked as mutually exclusive classes. The harmonized external counts exact/strong+/moderate+ are InvDesRoute 64/79/110; CSLLM 56/63/94; ChemLLM 58/66/99; ChemDFM 41/45/57; LLM4SYN 22/25/38. Invalid-output targets remain in the denominator.

The 145-record historical policy is not the same policy as the development-selected corpus/MMR ranking. Its template-first diverse display is preserved. Later-target exact/strong/moderate recovery is 47/49/53 under the archived policy. Removing the matching-template bonus yields 47/48/52. Full methods, coefficients and source records are in the data archive.
