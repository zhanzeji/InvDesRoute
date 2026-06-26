# Synthesis Pathway Screening

This is the rebuilt workflow for the actual target task:

> User inputs a compound formula or a CIF file. The code retrieves relevant phases from the local library and/or Materials Project, collects CIF files, optionally computes PET-OAM-XI energies, enumerates balanced reactions that form the target, computes reaction energies, analyzes phase stability / energy above hull, and reports likely synthesis pathways.

## Scope

- Works directly for systems covered by the local phase library.
- For arbitrary inorganic compounds such as `KI`, use `--database mp` or `--database local+mp` with a Materials Project API key.
- MP retrieval expands the target chemical system into all sub-chemical systems, so a target such as `KI` pulls `K`, `I`, and `K-I` phases rather than only the target phase.
- PET-OAM-XI reaction energies require CIF files for all phases and `--compute-pet`.
- Ehull is reported from MP when available; local DFT/PET hull diagnostics are also computed from the retrieved phase pool when enough energies are available.
- Reaction energies are ranked per target formula unit. MP reference reaction energies now use only `MP_FORM`, i.e. MP `formation_energy_per_atom` multiplied by formula-unit atom count and then summed over the balanced reaction.
- In addition to the purely thermodynamic best path, the report marks practical routes using low-complexity, low-Ehull phase filters.
- Practical-route and recommendation thresholds are configurable from the command line, so sensitivity tests can be run without editing source code.
- For literature-facing synthesis planning, the workflow now supports common experimental precursor expansion, gas/reservoir species, targeted precursor-route enumeration, virtual precursor placeholders, and a synthesizability-aware recommendation layer.

## Main command

```bash
python \
  <InvDesRoute_repo>/run_screening.py \
  --target-formula NaAlH4 \
  --database local \
  --output-dir <InvDesRoute_repo>/results/NaAlH4_local
```

For a general compound that is not in the local hydride library, e.g. KI:

```bash
export MP_API_KEY=your_mp_key
python \
  <InvDesRoute_repo>/run_screening.py \
  --target-formula KI \
  --database mp \
  --mp-max-phases 120 \
  --output-dir <InvDesRoute_repo>/results/KI_mp
```

With PET-OAM-XI energies:

```bash
python \
  <InvDesRoute_repo>/run_screening.py \
  --target-formula KI \
  --database mp \
  --compute-pet \
  --pet-device cuda \
  --output-dir <InvDesRoute_repo>/results/KI_mp_pet
```

The practical-route filters can be adjusted explicitly:

```bash
python \
  <InvDesRoute_repo>/run_screening.py \
  --target-formula KI \
  --database mp \
  --compute-pet \
  --practical-max-atoms 10 \
  --practical-max-stoich-ratio 5 \
  --practical-max-ehull 0.10 \
  --practical-max-coeff-sum-per-target 30 \
  --recommendation-exergonic-threshold -0.05 \
  --recommendation-ehull-threshold 0.10 \
  --output-dir <InvDesRoute_repo>/results/KI_mp_pet
```

For synthesis-facing recommendations that include common experimental precursors and route-prior scoring:

```bash
export MP_API_KEY=your_mp_key
<InvDesRoute_repo>/run_screening.py \
  --target-formula BaTiO3 \
  --database mp \
  --include-common-precursors \
  --include-gas-reservoirs \
  --include-targeted-precursor-routes \
  --byproducts-reservoir-only \
  --practical-max-atoms 20 \
  --practical-max-coeff-sum-per-target 80 \
  --output-dir <InvDesRoute_repo>/results/BaTiO3_synthesis_prior
```

For cases where MP lacks an experimental precursor, such as oxalates or hydrates, add virtual placeholders or a vetted manual thermochemistry table:

```bash
<InvDesRoute_repo>/run_screening.py \
  --target-formula LiFePO4 \
  --database mp \
  --include-common-precursors \
  --include-gas-reservoirs \
  --include-virtual-common-precursors \
  --manual-precursor-table /path/to/manual_precursor_table.csv \
  --include-targeted-precursor-routes \
  --byproducts-reservoir-only \
  --output-dir <InvDesRoute_repo>/results/LiFePO4_synthesis_prior
```

For a user CIF:

```bash
python \
  <InvDesRoute_repo>/run_screening.py \
  --target-cif /path/to/target.cif \
  --database local+mp \
  --compute-pet \
  --output-dir <InvDesRoute_repo>/results/target_from_cif
```

## Outputs

Each run writes:

- `phase_pool_raw.csv`: retrieved and merged phases.
- `phase_stability.csv`: phase energies, MP Ehull if available, local DFT/PET Ehull diagnostics.
- `pet_energy_cache.csv`: PET-OAM-XI energy per formula unit if `--compute-pet` is used.
- `formation_reactions.csv`: balanced candidate reactions forming the target.
- `decomposition_reactions.csv`: candidate target decomposition reactions.
- `report.md`: concise pathway and stability interpretation.
- `recommended_routes.csv`: ranked synthesis-facing route recommendations, using PET when available and falling back to `MP_FORM` only when PET energies are absent.
- `synthesis_recommendation_summary.csv`: one-row target-level classification, confidence score, risk flags, best PET/`MP_FORM` paths, and decomposition competition.
- `synthesis_recommendation.md`: user-facing synthesis recommendation with practical route, confidence, and next-step guidance.
- `synthesizability_ranked_routes.csv`: synthesis-prior route ranking that combines reaction energy, experimental route templates, precursor commonness, route complexity, phase availability, and condition-risk flags.
- `synthesizability_recommendation.md`: Chinese route recommendation report explaining why top routes are suggested and what risks remain.
- `run_metadata.json`: run settings and counts.

## Synthesizability-aware recommendation layer

The `synthesizability_ranked_routes.csv` output is deliberately different from pure reaction-energy ranking. It keeps the raw energy columns but adds:

- `synth_score`: combined synthesis-prior score.
- `route_template_score` and `route_template_reason`: whether the route matches common experimental families such as carbonate+oxide ceramic synthesis, lithium carbonate + transition-metal oxide cathode synthesis, ammonolysis/nitridation, carbonation, all-oxide solid-state synthesis, or direct elemental combination.
- `precursor_prior_score`: whether reactants are common experimental precursors, rare computed phases, reactive halogen/nitrogen species, gas reservoirs, or virtual precursor placeholders.
- `synthesis_strengths`: concise reasons for recommending the route.
- `synthesis_risk_flags`: issues such as gas release, atmosphere control, uphill static energy, missing virtual precursor thermochemistry, or failure of the original practical-route filter.

This layer is intended to answer: “What should an experimentalist inspect first?” It is not a replacement for finite-temperature thermodynamics, pressure/atmosphere optimization, DFT refinement, or experiment.

## Manual precursor thermochemistry

The template `data/manual_precursor_thermochemistry_template.csv` provides the format for adding vetted precursor formation energies. The corresponding explanation is in `docs/MANUAL_PRECURSOR_THERMOCHEMISTRY.md`.

Current policy:

- Do not invent unknown precursor energies.
- Use virtual precursors for route enumeration only when MP lacks a common experimental precursor.
- Treat `virtual_precursor_energy_missing` and `energy_unavailable` as refinement flags.
- Add temperature/pressure/gas-chemical-potential corrections only after route families are identified.

## PET/UPET 调用方式

当前工程使用的 PET 能量计算方式与历史脚本 `historical PET energy script` 保持一致：通过 `$UPET_PYTHON` 启动独立子进程，设置 `VESIN_CUDA_MAX_PAIRS_PER_POINT=4096`，加载 `upet.calculator.UPETCalculator`，并使用 `$PET_OAM_XL_CHECKPOINT` 作为 checkpoint。主命令可通过 `--pet-python` 显式指定 UPET 环境：

```bash
python \
  <InvDesRoute_repo>/run_screening.py \
  --target-formula KI \
  --database mp \
  --compute-pet \
  --pet-device auto \
  --pet-python $UPET_PYTHON \
  --output-dir <InvDesRoute_repo>/results/KI_mp_pet
```

## Benchmark

The current paper-scale MP-expanded PET benchmark covers 120 diverse non-hydride targets:

```bash
export MP_API_KEY=your_mp_key
python \
  <InvDesRoute_repo>/benchmark_mp_pet.py \
  --target-set subjournal120 \
  --output-dir <InvDesRoute_repo>/results/mp_pet_benchmark_120 \
  --mp-max-phases 20 \
  --pet-max-phases 25 \
  --pet-device auto \
  --pet-python $UPET_PYTHON \
  --keep-going
```

Latest paper-scale benchmark artifacts are in `results/mp_pet_benchmark_120/`:

- `mp_pet_benchmark_summary.csv`: target-level workflow and PET-vs-MP statistics.
- `mp_pet_reaction_data.csv`: pooled reaction-level PET and `MP_FORM` reaction energies.
- `mp_pet_benchmark_summary.png` / `.pdf`: summary figure.
- `README_summary.md`: short aggregate statistics.
- `synthesis_recommendation_summary_all.csv`: target-level recommendation summary.
- `recommended_routes_top5_all.csv`: top recommended routes for each target.
- `recommendation_category_summary.csv`: category-level recommendation statistics.
- `top1_route_feature_summary.csv`: precursor-complexity statistics for the top-1 recommended routes.
- `dft_refinement_queue.csv`: targets/routes that should be prioritized for DFT refinement or manual checking.
- `RECOMMENDATION_LAYER_ANALYSIS.md`: Chinese summary of the recommendation layer.
- `EXTENDED_PAPER_ANALYSIS.md`: additional uncertainty, top-5-overlap, precursor-complexity, and DFT-queue statistics used in the final manuscript.

Current aggregate result: 120/120 targets completed, 1619 retrieved phases, 1596 PET-evaluated phases, 18063 enumerated formation reactions, 17085 comparable PET/`MP_FORM` reactions, 4235 practical-route candidates, mean PET-vs-`MP_FORM` MAE of 0.584 eV/target, median MAE of 0.434 eV/target, mean sign consistency of 0.931, and top-1 pathway agreement of 0.825.

The recommendation layer classifies the same 120 targets into 104 high-confidence promising synthesis candidates, 15 medium-confidence candidates requiring DFT/experimental filtering, and 1 low-confidence case under the current screened network. These labels are heuristic synthesis-screening outputs based on target Ehull, exergonic practical routes, PET/`MP_FORM` pathway agreement, and decomposition competition. The aggregation script also produces a 26-target `dft_refinement_queue.csv` for cases with medium/low confidence, decomposition competition, or PET/`MP_FORM` pathway disagreement. The top-1 route feature summary shows that 65 targets use elemental precursors, 50 use simple compound precursors, and only 5 require complex precursor routes under the current PET-ranked practical-route layer.

To rebuild the recommendation summary from existing per-target folders:

```bash
python \
  <InvDesRoute_repo>/aggregate_recommendations.py \
  --benchmark-dir <InvDesRoute_repo>/results/mp_pet_benchmark_120 \
  --top-n 5
```

## 文献路线验证与 PET-aligned 合成性推荐

新增的 seed 文献路线验证用于回答：workflow 推荐的路线是否接近真实实验常用路线，而不只是给出热力学最低能反应。

批量生成 40 条文献 seed 目标：

```bash
python \
  <InvDesRoute_repo>/literature_benchmark/run_seed_synthesizability_cases.py \
  --mp-api-key "$MP_API_KEY" \
  --output-dir <InvDesRoute_repo>/results/seed40_synthesizability_cases
```

将已有 PET 固体能量缓存转换为 formation-energy-like 标尺，并重新排序：

```bash
python \
  <InvDesRoute_repo>/literature_benchmark/apply_pet_form_aligned_energy.py \
  --case-dir <InvDesRoute_repo>/results/seed40_synthesizability_cases
```

评估文献反应式和路线族匹配：

```bash
python \
  <InvDesRoute_repo>/literature_benchmark/evaluate_seed_synthesizability.py \
  --case-dir <InvDesRoute_repo>/results/seed40_synthesizability_cases \
  --outdir <InvDesRoute_repo>/literature_benchmark/results/seed40_synthesizability_pet_aligned
```

当前 PET-aligned 结果：

- 40/40 文献路线被枚举。
- 精确文献式 top-1：35/40；top-5：37/40；top-20：39/40。
- 路线族 top-5：39/40；top-20：40/40。
- 主要失败模式是 LiFePO4 的 FeC2O4/NH4H2PO4 缺可靠热化学或 PET 能量。

重要文件：

- `literature_benchmark/results/seed40_synthesizability_pet_aligned/SEED40_SYNTHESIZABILITY_VALIDATION.md`
- `literature_benchmark/results/seed40_synthesizability_pet_aligned/PET_ALIGNED_ENERGY_NOTES.md`
- `literature_benchmark/results/seed40_synthesizability_pet_aligned/seed40_validation_overview.svg`
- `results/seed40_synthesizability_cases/pet_form_aligned_energy_summary.csv`

## Design notes

This project is intentionally focused on the current paper task: arbitrary formula/CIF input, database phase retrieval, PET-OAM-XI energy evaluation, reaction-energy ranking, recommendation-layer classification, and Ehull-aware synthesis-path analysis.
