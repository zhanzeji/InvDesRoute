# InvDesRoute v0.2.0

InvDesRoute enumerates balanced inorganic reactions and ranks candidate synthesis hypotheses using interatomic-potential energies, precursor rules, a historical precursor prior and diversity-aware selection.

This release replaces the archived retrospective ranking with the same frozen policy used in the publication-time evaluation. It includes the revised 145-record ranking, evaluation-template controls, common-candidate AUC analysis, and 80-route positive-control reranking. Earlier versions remain separate records.

## Reproduce the published derived-data snapshot

Download and extract the code and data ZIPs into the same parent directory. Use Python 3.10 in a separate environment:

```sh
python -m pip install -r requirements.txt
python scripts/reproduce_public.py --data ../InvDesRoute-data-v0.2.0 --output recomputed/v0.2.0
```

The command verifies the data checksums, recomputes base scores, historical-prior mixture scores and MMR displays, and calculates literature recovery, zero-template-bonus recovery and common-subset AUC. It also aggregates the released hydride residuals, temporal records, external-model records and network metrics. Results and pass/fail checks are written to the output directory. No API key, model download or GPU is required for this path.

Candidate-reference similarities and reference-identity indicators are supplied as frozen evaluation outputs. This public path does not re-extract literature labels, rematch raw structures or rerun MLIP/DFT calculations. Hydride errors reproduce an overlapping-reference reconstruction analysis; they do not establish held-out energy accuracy.

## Frozen recommendation policy

Component order is energy, rule-based precursor prior, template, practical status, complexity, phase availability and condition simplicity. Weights are `(0.10, 0.22, 0.22, 0.08, 0.08, 0.10, 0.20)`. The ranking template bonus is retained and the base score is capped at one. The historical prior contributes 0.30 and the MMR penalty is 0.10. The similarity metric is an evaluation endpoint, not a recommendation score.

The primary retrospective denominator remains 145 cohort records, including a FeS reference appearing in two source cohorts. There are 144 distinct target/DOI/precursor/byproduct keys. The deduplicated table is sensitivity only. Primary top-five recovery is 62 strict identities, 92 strong-or-complete-recall matches and 115 moderate-or-better matches.

## Recompute from separately obtained source-dependent inputs

```sh
python scripts/reproduce_full.py --inputs /path/to/InvDesRoute-private-inputs-v0.2.0 --output recomputed/source-dependent
```

This path rebuilds the historical prior from its archived records and recomputes the 145-record matches, paired template-bonus controls, common-candidate discrimination and the 80-route control. The corresponding complete input archive is retained separately because redistribution conditions for some source records have not been established. Obtain the original data from their providers or request the documented input manifest from the authors. Do not infer that the public archive alone reproduces fresh structure retrieval, MLIP inference, source-label extraction or every calculation in the manuscript.

## Repository contents

| Directory | Contents |
|---|---|
| `scripts/` | Current public and source-dependent reproduction entry points |
| `analysis/` | Revised scoring and numerical-analysis scripts |
| `code/workflow/` | Workflow modules used by the analyses; historical helper names may retain legacy terminology |
| `docs/` | Data index, interpretation limits, release changes and source attribution |
| `environment-tested.json` | Versions used for the release checks |

Original code is released under MIT. Third-party libraries and model checkpoints retain their own terms. Model weights, raw MP structures, raw literature recipes and raw JANAF tables are not distributed in this public release. The official EquiformerV3 implementation and weights are available from https://github.com/atomicarchitects/equiformer_v3 .

See `docs/RESULTS_AND_LIMITATIONS.md` before interpreting benchmark accuracy or an individual synthesis recommendation.
