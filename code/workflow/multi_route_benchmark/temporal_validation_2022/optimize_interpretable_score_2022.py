#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
BENCHMARK = PROJECT / "multi_route_benchmark"
LITERATURE = PROJECT / "literature_benchmark"
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(LITERATURE))

from evaluate_route_similarity import parse_side, route_similarity, split_formulas  # noqa: E402
from optimize_ranking_target_disjoint import CorpusPrior, condition_heavy_score, route_redundancy  # noqa: E402


COMPONENTS = [
    "energy_score",
    "precursor_prior_score",
    "route_template_score",
    "practical_filter_score",
    "route_complexity_score",
    "phase_availability_score",
    "condition_simplicity_score",
]
PHYSICAL_PRIOR = np.asarray([0.10, 0.25, 0.25, 0.10, 0.07, 0.08, 0.15], dtype=float)
MINIMUM = np.asarray([0.07, 0.12, 0.12, 0.04, 0.04, 0.04, 0.06], dtype=float)
MAXIMUM = np.asarray([0.28, 0.35, 0.35, 0.20, 0.18, 0.18, 0.25], dtype=float)


@dataclass
class TargetData:
    routes: list[dict[str, object]]
    components: np.ndarray
    corpus_scores: np.ndarray
    references: list[dict[str, object]]
    similarity: np.ndarray
    precursor_recall: np.ndarray
    strict_exact: np.ndarray
    strong: np.ndarray
    moderate: np.ndarray


def target_fold(target: str) -> int:
    return int(hashlib.sha256(target.encode()).hexdigest()[:8], 16) % 5


def load_case(target: str, roots: list[Path]) -> list[dict[str, object]]:
    for root in roots:
        path = root / target / "synthesizability_ranked_routes.csv"
        if path.exists():
            return pd.read_csv(path).replace({np.nan: ""}).to_dict("records")
    raise FileNotFoundError(f"No route network for {target}")


def prepare_targets(references: pd.DataFrame, roots: list[Path], prior: CorpusPrior) -> dict[str, TargetData]:
    prepared = {}
    for target, target_refs in references.groupby("target"):
        routes = load_case(target, roots)
        components = np.asarray(
            [[float(route.get(component) or 0.0) for component in COMPONENTS] for route in routes], dtype=float
        )
        corpus = np.asarray([prior.score(target, parse_side(route.get("reactants"))) for route in routes])
        similarities, recalls, exacts, strongs, moderates = [], [], [], [], []
        ref_rows = target_refs.to_dict("records")
        for reference in ref_rows:
            lit_precursors = split_formulas(reference.get("literature_precursors"))
            lit_byproducts = split_formulas(reference.get("literature_byproducts"))
            sim_row, recall_row, exact_row, strong_row, moderate_row = [], [], [], [], []
            for route in routes:
                result = route_similarity(target, lit_precursors, lit_byproducts, route)
                candidate_precursors = parse_side(route.get("reactants"))
                candidate_byproducts = parse_side(route.get("products")) - {target}
                strict = candidate_precursors == lit_precursors and candidate_byproducts == lit_byproducts
                score = float(result["route_similarity_score"])
                sim_row.append(score)
                recall_row.append(float(result["exact_reactant_recall"]))
                exact_row.append(strict)
                strong_row.append(strict or score >= 0.78)
                moderate_row.append(strict or score >= 0.62)
            similarities.append(sim_row)
            recalls.append(recall_row)
            exacts.append(exact_row)
            strongs.append(strong_row)
            moderates.append(moderate_row)
        prepared[target] = TargetData(
            routes=routes,
            components=components,
            corpus_scores=corpus,
            references=ref_rows,
            similarity=np.asarray(similarities),
            precursor_recall=np.asarray(recalls),
            strict_exact=np.asarray(exacts, dtype=bool),
            strong=np.asarray(strongs, dtype=bool),
            moderate=np.asarray(moderates, dtype=bool),
        )
    return prepared


def top_indices(data: TargetData, weights: np.ndarray, alpha: float, penalty: float, top_k: int) -> list[int]:
    base = data.components @ weights
    scores = (1.0 - alpha) * base + alpha * data.corpus_scores
    if penalty <= 0:
        return np.argsort(-scores, kind="stable")[:top_k].tolist()
    ordered = np.argsort(-scores, kind="stable").tolist()
    selected: list[int] = []
    while ordered and len(selected) < top_k:
        best_index, best_key = None, None
        for index in ordered:
            if best_key is not None and scores[index] < best_key[0]:
                break
            redundancy = max(
                (route_redundancy(data.references[0]["target"], data.routes[index], data.routes[chosen]) for chosen in selected),
                default=0.0,
            )
            key = (float(scores[index] - penalty * redundancy), float(scores[index]), -index)
            if best_key is None or key > best_key:
                best_index, best_key = index, key
        selected.append(int(best_index))
        ordered.remove(best_index)
    return selected


def evaluate(
    prepared: dict[str, TargetData],
    weights: np.ndarray,
    alpha: float,
    penalty: float,
    top_k: int = 5,
    targets: set[str] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    for target, data in prepared.items():
        if targets is not None and target not in targets:
            continue
        indices = top_indices(data, weights, alpha, penalty, top_k)
        for ref_index, reference in enumerate(data.references):
            local = data.similarity[ref_index, indices]
            best_local = int(np.argmax(local))
            route_index = indices[best_local]
            route = data.routes[route_index]
            rows.append(
                {
                    "route_id": reference.get("route_id", ""),
                    "target": target,
                    "doi": reference.get("doi", ""),
                    "publication_year": reference.get("publication_year", ""),
                    "matched_rank": best_local + 1,
                    "matched_reaction": route.get("reaction", ""),
                    "strict_exact": bool(data.strict_exact[ref_index, route_index]),
                    "strong_or_exact": bool(data.strong[ref_index, route_index]),
                    "moderate_or_better": bool(data.moderate[ref_index, route_index]),
                    "similarity": float(data.similarity[ref_index, route_index]),
                    "precursor_recall": float(data.precursor_recall[ref_index, route_index]),
                    "reaction_energy_eV_per_target": route.get("synth_rank_deltaE_per_target", ""),
                    "energy_source": route.get("synth_rank_energy_source", ""),
                    "risk_flags": route.get("synthesis_risk_flags", ""),
                    **{component: float(route.get(component) or 0.0) for component in COMPONENTS},
                }
            )
    detail = pd.DataFrame(rows)
    metrics = {
        "records": len(detail),
        "targets": int(detail["target"].nunique()),
        "strict_exact": int(detail["strict_exact"].sum()),
        "strong_or_exact": int(detail["strong_or_exact"].sum()),
        "moderate_or_better": int(detail["moderate_or_better"].sum()),
        "mean_similarity": float(detail["similarity"].mean()),
        "mean_precursor_recall": float(detail["precursor_recall"].mean()),
    }
    return metrics, detail


def objective(metrics: dict[str, float], weights: np.ndarray) -> float:
    records = metrics["records"]
    performance = (
        0.50 * metrics["strong_or_exact"] / records
        + 0.25 * metrics["strict_exact"] / records
        + 0.15 * metrics["moderate_or_better"] / records
        + 0.10 * metrics["mean_similarity"]
    )
    regularization = 0.02 * float(np.abs(weights - PHYSICAL_PRIOR).sum())
    return performance - regularization


def weight_candidates(seed: int = 20260911, count: int = 1500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = [PHYSICAL_PRIOR, np.ones(len(COMPONENTS)) / len(COMPONENTS)]
    while len(values) < count:
        concentration = PHYSICAL_PRIOR * 35 if len(values) % 2 else np.ones(len(COMPONENTS)) * 2
        candidate = rng.dirichlet(concentration)
        if np.all(candidate >= MINIMUM) and np.all(candidate <= MAXIMUM):
            values.append(candidate)
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select interpretable route-score weights before post-2022 testing.")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--future-case-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    development = pd.read_csv(args.split_dir / "parameter_development_routes.csv").fillna("")
    future = pd.read_csv(args.split_dir / "post2022_future_test_routes.csv").fillna("")
    prior_routes = pd.read_csv(args.split_dir / "pre2023_prior_routes.csv", low_memory=False).fillna("")
    prior = CorpusPrior.fit(prior_routes, set())
    dev_roots = [
        BENCHMARK / "cases_blind_equiformer_v3",
        BENCHMARK / "temporal_validation" / "cases_blind",
    ]
    future_roots = [args.future_case_root]
    dev_data = prepare_targets(development, dev_roots, prior)
    future_data = prepare_targets(future, future_roots, prior)

    candidates = weight_candidates()
    search_rows = []
    for index, weights in enumerate(candidates):
        metrics, _ = evaluate(dev_data, weights, 0.0, 0.0)
        search_rows.append(
            {
                "weight_id": index,
                **{component: weights[position] for position, component in enumerate(COMPONENTS)},
                **metrics,
                "objective": objective(metrics, weights),
                "l1_from_physical_prior": float(np.abs(weights - PHYSICAL_PRIOR).sum()),
            }
        )
    search = pd.DataFrame(search_rows).sort_values(
        ["objective", "strong_or_exact", "strict_exact", "moderate_or_better", "mean_similarity", "l1_from_physical_prior"],
        ascending=[False, False, False, False, False, True],
    )
    selected = search.iloc[0]
    weights = selected[COMPONENTS].to_numpy(float)

    cv_rows = []
    for fold in range(5):
        validation_targets = {target for target in dev_data if target_fold(target) == fold}
        training_targets = set(dev_data) - validation_targets
        fold_search = []
        for index, candidate in enumerate(candidates):
            metrics, _ = evaluate(dev_data, candidate, 0.0, 0.0, targets=training_targets)
            fold_search.append((objective(metrics, candidate), index))
        _, best_index = max(fold_search)
        metrics, _ = evaluate(dev_data, candidates[best_index], 0.0, 0.0, targets=validation_targets)
        cv_rows.append({"fold": fold, "selected_weight_id": best_index, **metrics})

    alpha_rows = []
    for alpha in np.arange(0.0, 0.41, 0.05):
        for penalty in (0.0, 0.04, 0.08, 0.10, 0.12, 0.16, 0.20):
            metrics, _ = evaluate(dev_data, weights, float(alpha), penalty)
            alpha_rows.append({"alpha": float(alpha), "lambda": penalty, **metrics})
    alpha_search = pd.DataFrame(alpha_rows).sort_values(
        ["strong_or_exact", "strict_exact", "moderate_or_better", "mean_similarity", "alpha", "lambda"],
        ascending=[False, False, False, False, True, True],
    )
    selected_alpha = float(alpha_search.iloc[0]["alpha"])
    selected_lambda = float(alpha_search.iloc[0]["lambda"])

    policies = []
    details = []
    energy_weights = np.zeros(len(COMPONENTS)); energy_weights[0] = 1.0
    for name, policy_weights, alpha, penalty in [
        ("energy_only", energy_weights, 0.0, 0.0),
        ("physical_prior", PHYSICAL_PRIOR, 0.0, 0.0),
        ("optimized_base", weights, 0.0, 0.0),
        ("optimized_plus_pre2023_prior", weights, selected_alpha, 0.0),
        ("optimized_plus_prior_and_mmr", weights, selected_alpha, selected_lambda),
    ]:
        metrics, detail = evaluate(future_data, policy_weights, alpha, penalty)
        policies.append({"policy": name, "alpha": alpha, "lambda": penalty, **metrics})
        detail.insert(0, "policy", name)
        details.append(detail)

    component_rows = []
    final_metrics = policies[-1]
    for position, component in enumerate(COMPONENTS):
        ablated = weights.copy(); ablated[position] = 0.0; ablated /= ablated.sum()
        metrics, _ = evaluate(future_data, ablated, selected_alpha, selected_lambda)
        component_rows.append(
            {
                "removed_component": component,
                **metrics,
                "delta_strong": final_metrics["strong_or_exact"] - metrics["strong_or_exact"],
                "delta_exact": final_metrics["strict_exact"] - metrics["strict_exact"],
                "delta_moderate": final_metrics["moderate_or_better"] - metrics["moderate_or_better"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    search.to_csv(args.output_dir / "historical_weight_search.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(args.output_dir / "historical_target_cv.csv", index=False)
    alpha_search.to_csv(args.output_dir / "historical_alpha_lambda_search.csv", index=False)
    pd.DataFrame(policies).to_csv(args.output_dir / "post2022_policy_ablation.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(args.output_dir / "post2022_route_evidence.csv", index=False)
    pd.DataFrame(component_rows).to_csv(args.output_dir / "post2022_component_ablation.csv", index=False)
    summary = {
        "weight_components": COMPONENTS,
        "physical_prior_weights": dict(zip(COMPONENTS, PHYSICAL_PRIOR.tolist())),
        "selected_weights": dict(zip(COMPONENTS, weights.tolist())),
        "weight_selection_records": int(len(development)),
        "weight_selection_targets": int(development["target"].nunique()),
        "selected_alpha": selected_alpha,
        "selected_lambda": selected_lambda,
        "future_records": int(len(future)),
        "future_targets": int(future["target"].nunique()),
        "future_policy_metrics": policies,
        "claim_boundary": "Weights and literature prior are selected without post-2022 route labels; MP and MLIP training data are not time frozen.",
    }
    (args.output_dir / "score_optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
