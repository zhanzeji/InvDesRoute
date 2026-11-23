#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
BENCHMARK = PROJECT / "multi_route_benchmark"
LITERATURE = PROJECT / "literature_benchmark"
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(LITERATURE))

from evaluate_route_similarity import (  # noqa: E402
    class_multiset,
    normalize,
    parse_side,
    route_similarity,
    split_formulas,
)
from optimize_ranking_target_disjoint import CorpusPrior, route_redundancy  # noqa: E402


COMPONENTS = (
    "energy_score",
    "precursor_prior_score",
    "route_template_score",
    "practical_filter_score",
    "route_complexity_score",
    "phase_availability_score",
    "condition_simplicity_score",
)
PROFILES = {
    "balanced": (0.15, 0.20, 0.20, 0.10, 0.10, 0.10, 0.15),
    "route_evidence": (0.10, 0.25, 0.25, 0.10, 0.07, 0.08, 0.15),
    "energy_heavy": (0.30, 0.20, 0.20, 0.08, 0.07, 0.10, 0.05),
    "precursor_heavy": (0.10, 0.35, 0.25, 0.08, 0.07, 0.05, 0.10),
    "template_heavy": (0.10, 0.20, 0.35, 0.08, 0.07, 0.05, 0.15),
    "practical_heavy": (0.12, 0.23, 0.23, 0.22, 0.06, 0.06, 0.08),
    "availability_heavy": (0.15, 0.20, 0.20, 0.10, 0.08, 0.17, 0.10),
    "condition_heavy": (0.10, 0.22, 0.22, 0.08, 0.08, 0.10, 0.20),
}
MODERATE_LABELS = {
    "exact_route_recovered",
    "strong_route_family_match",
    "moderate_route_family_match",
}
STRONG_LABELS = {"exact_route_recovered", "strong_route_family_match"}


def energy_component(delta_e: object) -> float:
    try:
        value = float(delta_e)
    except (TypeError, ValueError):
        return 0.35
    if not math.isfinite(value):
        return 0.35
    if value <= -1.0:
        return 1.00
    if value <= -0.05:
        return 0.88
    if value <= 0.25:
        return 0.64
    if value <= 1.0:
        return 0.42
    return 0.20


def profile_score(
    route: dict[str, object],
    weights: tuple[float, ...],
    energy_mode: str = "production",
    removed_component: str | None = None,
) -> float:
    values = {component: float(route.get(component) or 0.0) for component in COMPONENTS}
    if energy_mode == "mp_form":
        values["energy_score"] = energy_component(route.get("DeltaE_MP_FORM_per_target"))
    elif energy_mode == "no_energy":
        removed_component = "energy_score"
    retained = [(component, weight) for component, weight in zip(COMPONENTS, weights) if component != removed_component]
    weight_sum = sum(weight for _, weight in retained)
    score = sum(values[component] * weight / weight_sum for component, weight in retained)
    template = values["route_template_score"]
    if removed_component != "route_template_score":
        if template >= 0.95:
            score += 0.06
        elif template >= 0.90:
            score += 0.03
    return float(min(1.0, score))


def load_cases(case_root: Path, targets: set[str]) -> dict[str, list[dict[str, object]]]:
    cases = {}
    for target in sorted(targets):
        path = case_root / target / "synthesizability_ranked_routes.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        cases[target] = pd.read_csv(path).replace({np.nan: ""}).to_dict("records")
    return cases


def prepare_features(
    cases: dict[str, list[dict[str, object]]], prior: CorpusPrior
) -> dict[str, list[dict[str, object]]]:
    for target, routes in cases.items():
        for route in routes:
            route["historical_corpus_score"] = prior.score(target, parse_side(route.get("reactants")))
            route["profile_scores"] = {
                profile: profile_score(route, weights) for profile, weights in PROFILES.items()
            }
    return cases


def rank_routes(
    target: str,
    routes: list[dict[str, object]],
    profile: str,
    alpha: float,
    diversity_penalty: float,
    top_k: int,
    energy_mode: str = "production",
    removed_component: str | None = None,
    energy_only: bool = False,
) -> list[dict[str, object]]:
    weights = PROFILES[profile]
    candidates = []
    for original in routes:
        route = dict(original)
        if energy_only:
            value = route.get("synth_rank_deltaE_per_target")
            try:
                rank_score = -float(value)
            except (TypeError, ValueError):
                rank_score = -float("inf")
            base_score = energy_component(value)
        else:
            base_score = profile_score(route, weights, energy_mode, removed_component)
            rank_score = (1.0 - alpha) * base_score + alpha * float(route["historical_corpus_score"])
        route["evaluation_base_score"] = base_score
        route["evaluation_rank_score"] = rank_score
        candidates.append(route)

    remaining = sorted(candidates, key=lambda route: (-float(route["evaluation_rank_score"]), str(route.get("reaction", ""))))
    if diversity_penalty <= 0 or energy_only:
        return remaining[:top_k]
    selected: list[dict[str, object]] = []
    while remaining and len(selected) < top_k:
        best = None
        best_key = None
        for route in remaining:
            score = float(route["evaluation_rank_score"])
            if best_key is not None and score < best_key[0]:
                break
            utility = score - diversity_penalty * max(
                (route_redundancy(target, route, chosen) for chosen in selected), default=0.0
            )
            key = (utility, score, str(route.get("reaction", "")))
            if best_key is None or key > best_key:
                best = route
                best_key = key
        assert best is not None
        selected.append(best)
        remaining.remove(best)
    return selected


def evaluate(
    references: pd.DataFrame,
    cases: dict[str, list[dict[str, object]]],
    profile: str,
    alpha: float,
    diversity_penalty: float,
    top_k: int = 5,
    energy_mode: str = "production",
    removed_component: str | None = None,
    energy_only: bool = False,
) -> tuple[dict[str, object], pd.DataFrame]:
    rows = []
    ranked_cache = {
        target: rank_routes(
            target,
            cases[target],
            profile,
            alpha,
            diversity_penalty,
            top_k,
            energy_mode,
            removed_component,
            energy_only,
        )
        for target in references["target"].unique()
    }
    for _, reference in references.iterrows():
        target = reference["target"]
        lit_precursors = split_formulas(reference["literature_precursors"])
        lit_byproducts = split_formulas(reference["literature_byproducts"])
        best = None
        for rank, route in enumerate(ranked_cache[target], start=1):
            similarity = route_similarity(target, lit_precursors, lit_byproducts, route)
            candidate_precursors = parse_side(route.get("reactants"))
            candidate_byproducts = parse_side(route.get("products")) - {target}
            strict_exact = candidate_precursors == lit_precursors and candidate_byproducts == lit_byproducts
            key = (strict_exact, similarity["route_similarity_score"], -rank)
            if best is None or key > best[0]:
                best = (key, rank, route, similarity, strict_exact)
        _, rank, route, similarity, strict_exact = best
        rows.append(
            {
                "route_id": reference["route_id"],
                "target": target,
                "doi": reference["doi"],
                "publication_year": reference["publication_year"],
                "material_category": reference["material_category"],
                "profile": profile,
                "alpha": alpha,
                "diversity_penalty": diversity_penalty,
                "top_k": top_k,
                "energy_mode": "energy_only" if energy_only else energy_mode,
                "strict_exact": strict_exact,
                "strong_or_exact": similarity["route_similarity_label"] in STRONG_LABELS,
                "moderate_or_better": similarity["route_similarity_label"] in MODERATE_LABELS,
                "best_rank": rank,
                "matched_reaction": route.get("reaction", ""),
                "matched_rank_score": route.get("evaluation_rank_score", ""),
                "matched_base_score": route.get("evaluation_base_score", ""),
                "historical_corpus_score": route.get("historical_corpus_score", ""),
                "production_delta_e_eV_per_target": route.get("synth_rank_deltaE_per_target", ""),
                "mp_form_delta_e_eV_per_target": route.get("DeltaE_MP_FORM_per_target", ""),
                "energy_source": route.get("synth_rank_energy_source", ""),
                "risk_flags": route.get("synthesis_risk_flags", ""),
                **similarity,
            }
        )
    detail = pd.DataFrame(rows)
    displayed = [route for target_routes in ranked_cache.values() for route in target_routes]
    displayed_energy = pd.to_numeric(
        pd.Series([route.get("synth_rank_deltaE_per_target") for route in displayed]),
        errors="coerce",
    )
    targets_with_exergonic = 0
    for target_routes in ranked_cache.values():
        energies = pd.to_numeric(
            pd.Series([route.get("synth_rank_deltaE_per_target") for route in target_routes]),
            errors="coerce",
        )
        targets_with_exergonic += int(energies.lt(-0.05).any())
    metrics = {
        "records": len(detail),
        "targets": int(detail["target"].nunique()),
        "strict_exact": int(detail["strict_exact"].sum()),
        "strong_or_exact": int(detail["strong_or_exact"].sum()),
        "moderate_or_better": int(detail["moderate_or_better"].sum()),
        "mean_similarity": float(detail["route_similarity_score"].mean()),
        "mean_precursor_recall": float(detail["exact_reactant_recall"].mean()),
        "displayed_routes": len(displayed),
        "displayed_exergonic_routes": int(displayed_energy.lt(-0.05).sum()),
        "displayed_strongly_uphill_routes": int(displayed_energy.gt(1.0).sum()),
        "targets_with_exergonic_topk": targets_with_exergonic,
    }
    return metrics, detail


def objective(metrics: dict[str, object], complexity: tuple[object, ...]) -> tuple[object, ...]:
    return (
        int(metrics["strong_or_exact"]),
        int(metrics["strict_exact"]),
        int(metrics["moderate_or_better"]),
        float(metrics["mean_similarity"]),
        *complexity,
    )


def select_parameters(
    references: pd.DataFrame,
    cases: dict[str, list[dict[str, object]]],
) -> tuple[str, float, float, pd.DataFrame]:
    rows = []
    for profile in PROFILES:
        for alpha in (0.0, 0.1, 0.2, 0.3, 0.4):
            metrics, _ = evaluate(references, cases, profile, alpha, 0.0)
            rows.append({"stage": "profile_and_prior", "profile": profile, "alpha": alpha, "lambda": 0.0, **metrics})
    first = max(
        rows,
        key=lambda row: objective(row, (-float(row["alpha"]), -list(PROFILES).index(row["profile"]))),
    )
    profile, alpha = str(first["profile"]), float(first["alpha"])
    for penalty in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        metrics, _ = evaluate(references, cases, profile, alpha, penalty)
        rows.append({"stage": "diversity", "profile": profile, "alpha": alpha, "lambda": penalty, **metrics})
    second_rows = [row for row in rows if row["stage"] == "diversity"]
    second = max(second_rows, key=lambda row: objective(row, (-float(row["lambda"]),)))
    return profile, alpha, float(second["lambda"]), pd.DataFrame(rows)


def cross_validate(
    references: pd.DataFrame,
    cases: dict[str, list[dict[str, object]]],
    folds: int = 5,
) -> pd.DataFrame:
    targets = references[["target", "material_category"]].drop_duplicates().copy()
    rng = np.random.default_rng(20220912)
    fold_map: dict[str, int] = {}
    for _, group in targets.groupby("material_category"):
        values = list(group["target"])
        rng.shuffle(values)
        for index, target in enumerate(values):
            fold_map[target] = index % folds
    rows = []
    for fold in range(folds):
        train = references.loc[references["target"].map(fold_map).ne(fold)]
        test = references.loc[references["target"].map(fold_map).eq(fold)]
        profile, alpha, penalty, _ = select_parameters(train, cases)
        metrics, _ = evaluate(test, cases, profile, alpha, penalty)
        rows.append({"fold": fold, "profile": profile, "alpha": alpha, "lambda": penalty, **metrics})
    return pd.DataFrame(rows)


def thermodynamic_consistency(case_root: Path, targets: set[str]) -> tuple[dict[str, object], pd.DataFrame]:
    rows = []
    for target in sorted(targets):
        frame = pd.read_csv(case_root / target / "formation_reactions.csv")
        model_col = "DeltaE_PRODUCTION_FORM_ALIGNED_per_target"
        reference_col = "DeltaE_MP_FORM_per_target"
        valid = frame.dropna(subset=[model_col, reference_col]).copy()
        sign = ((valid[model_col] * valid[reference_col]) > 0) | (
            valid[model_col].abs().lt(1e-12) & valid[reference_col].abs().lt(1e-12)
        )
        model_top = valid.nsmallest(min(5, len(valid)), model_col)
        reference_top = valid.nsmallest(min(5, len(valid)), reference_col)
        rows.append(
            {
                "target": target,
                "comparable_routes": len(valid),
                "mae_eV_per_target": float((valid[model_col] - valid[reference_col]).abs().mean()),
                "sign_consistency": float(sign.mean()),
                "top1_agreement": bool(len(valid) and model_top.iloc[0]["reaction"] == reference_top.iloc[0]["reaction"]),
                "top5_overlap": len(set(model_top["reaction"]) & set(reference_top["reaction"])),
            }
        )
    detail = pd.DataFrame(rows)
    summary = {
        "targets": len(detail),
        "comparable_routes": int(detail["comparable_routes"].sum()),
        "macro_mae_eV_per_target": float(detail["mae_eV_per_target"].mean()),
        "macro_sign_consistency": float(detail["sign_consistency"].mean()),
        "top1_agreement": int(detail["top1_agreement"].sum()),
        "mean_top5_overlap": float(detail["top5_overlap"].mean()),
    }
    return summary, detail


def impurity_audit(references: pd.DataFrame, case_root: Path) -> tuple[dict[str, object], pd.DataFrame]:
    rows = []
    for _, reference in references.loc[references["has_reported_impurity"].astype(bool)].iterrows():
        target = reference["target"]
        impurities = split_formulas(reference["impurity_phases"])
        phases = pd.read_csv(case_root / target / "phase_stability.csv").fillna("")
        phase_formulas = {normalize(value) for value in phases["formula"]}
        decomposition_path = case_root / target / "decomposition_reactions.csv"
        decompositions = pd.read_csv(decomposition_path).fillna("") if decomposition_path.exists() else pd.DataFrame()
        decomposition_products = set()
        impurity_decomposition_energies = []
        if len(decompositions):
            for _, decomposition in decompositions.iterrows():
                products = parse_side(decomposition["products"])
                decomposition_products.update(products)
                if products & impurities:
                    try:
                        impurity_decomposition_energies.append(
                            float(decomposition["DeltaE_PRODUCTION_FORM_ALIGNED_per_target"])
                        )
                    except (KeyError, TypeError, ValueError):
                        pass
        energy_sources = pd.read_csv(
            case_root / target / "production_form_aligned_phase_energy_sources.csv"
        ).fillna("")
        energy_map = {
            normalize(row["formula"]): row["PRODUCTION_FORM_ALIGNED_source"]
            for _, row in energy_sources.iterrows()
            if row.get("PRODUCTION_FORM_ALIGNED_energy_per_formula", "") != ""
        }
        rows.append(
            {
                "route_id": reference["route_id"],
                "target": target,
                "doi": reference["doi"],
                "reported_impurities": ";".join(sorted(impurities)),
                "reported_impurity_count": len(impurities),
                "phase_pool_impurities": ";".join(sorted(impurities & phase_formulas)),
                "phase_pool_recall": len(impurities & phase_formulas) / len(impurities) if impurities else 1.0,
                "energy_resolved_impurities": ";".join(sorted(impurities & set(energy_map))),
                "energy_resolved_recall": len(impurities & set(energy_map)) / len(impurities) if impurities else 1.0,
                "decomposition_impurities": ";".join(sorted(impurities & decomposition_products)),
                "decomposition_recall": len(impurities & decomposition_products) / len(impurities) if impurities else 1.0,
                "best_impurity_decomposition_delta_e_eV_per_target": min(impurity_decomposition_energies) if impurity_decomposition_energies else math.nan,
            }
        )
    detail = pd.DataFrame(rows)
    summary = {
        "routes_with_reported_impurities": len(detail),
        "reported_impurity_formulas": int(detail["reported_impurity_count"].sum()) if len(detail) else 0,
        "mean_phase_pool_recall": float(detail["phase_pool_recall"].mean()) if len(detail) else math.nan,
        "mean_energy_resolved_recall": float(detail["energy_resolved_recall"].mean()) if len(detail) else math.nan,
        "mean_decomposition_recall": float(detail["decomposition_recall"].mean()) if len(detail) else math.nan,
    }
    return summary, detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the 2022-cutoff InvDesRoute benchmark.")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    development = pd.read_csv(args.split_dir / "development_routes_2021_2022.csv").fillna("")
    future = pd.read_csv(args.split_dir / "future_routes_2023_2024.csv").fillna("")
    prior_frame = pd.read_csv(args.split_dir / "historical_prior_routes_through_2020.csv").fillna("")
    targets = set(development["target"]) | set(future["target"])
    prior = CorpusPrior.fit(prior_frame, targets)
    cases = prepare_features(load_cases(args.case_root, targets), prior)

    cv = cross_validate(development, cases)
    profile, alpha, penalty, search = select_parameters(development, cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv.to_csv(args.output_dir / "development_nested_cv.csv", index=False)
    search.to_csv(args.output_dir / "development_parameter_search.csv", index=False)

    ablations = []
    details = []
    configurations = [
        ("production_energy_only", "production", None, True),
        ("selected_without_energy", "no_energy", None, False),
        ("selected_with_mp_form_energy", "mp_form", None, False),
        ("selected_with_equiformer_energy", "production", None, False),
    ]
    for label, energy_mode, removed, energy_only in configurations:
        for top_k in (1, 5, 10, 20):
            metrics, detail = evaluate(
                future,
                cases,
                profile,
                alpha,
                penalty,
                top_k,
                energy_mode,
                removed,
                energy_only,
            )
            ablations.append({"configuration": label, "top_k": top_k, **metrics})
            detail["configuration"] = label
            details.append(detail)

    component_rows = []
    for component in COMPONENTS:
        metrics, _ = evaluate(
            future, cases, profile, alpha, penalty, 5, "production", component, False
        )
        component_rows.append({"removed_component": component, **metrics})

    final_metrics, final_detail = evaluate(future, cases, profile, alpha, penalty, 5)
    full_rows = []
    for _, reference in future.iterrows():
        target = reference["target"]
        lit_precursors = split_formulas(reference["literature_precursors"])
        lit_byproducts = split_formulas(reference["literature_byproducts"])
        best = max(
            (route_similarity(target, lit_precursors, lit_byproducts, route) for route in cases[target]),
            key=lambda value: value["route_similarity_score"],
        )
        top = final_detail.loc[final_detail["route_id"].eq(reference["route_id"])].iloc[0]
        full_rows.append(
            {
                "route_id": reference["route_id"],
                "target": target,
                "publication_year": reference["publication_year"],
                "literature_precursor_classes": ";".join(sorted(class_multiset(lit_precursors, target).elements())),
                "full_network_score": best["route_similarity_score"],
                "full_network_label": best["route_similarity_label"],
                "top5_score": top["route_similarity_score"],
                "top5_label": top["route_similarity_label"],
                "diagnosis": (
                    "candidate_space_gap"
                    if best["route_similarity_label"] not in MODERATE_LABELS
                    else "ranking_gap"
                    if not bool(top["moderate_or_better"])
                    else "specificity_gap"
                    if best["route_similarity_label"] in STRONG_LABELS and not bool(top["strong_or_exact"])
                    else "recovered"
                ),
            }
        )
    failures = pd.DataFrame(full_rows)

    consistency_summary, consistency_detail = thermodynamic_consistency(args.case_root, set(future["target"]))
    impurity_summary, impurity_detail = impurity_audit(future, args.case_root)

    pd.DataFrame(ablations).to_csv(args.output_dir / "future_ranking_and_energy_ablation.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(args.output_dir / "future_route_evidence.csv", index=False)
    pd.DataFrame(component_rows).to_csv(args.output_dir / "future_component_ablation.csv", index=False)
    failures.to_csv(args.output_dir / "future_failure_diagnostics.csv", index=False)
    consistency_detail.to_csv(args.output_dir / "future_equiformer_mp_consistency_by_target.csv", index=False)
    impurity_detail.to_csv(args.output_dir / "future_reported_impurity_audit.csv", index=False)

    summary = {
        "selected_profile": profile,
        "selected_weights": dict(zip(COMPONENTS, PROFILES[profile])),
        "selected_historical_corpus_weight": alpha,
        "selected_diversity_penalty": penalty,
        "development_nested_cv": {
            "folds": len(cv),
            "records": int(cv["records"].sum()),
            "strong_or_exact": int(cv["strong_or_exact"].sum()),
            "moderate_or_better": int(cv["moderate_or_better"].sum()),
            "strict_exact": int(cv["strict_exact"].sum()),
            "profile_selection_counts": dict(Counter(cv["profile"])),
        },
        "future_top5": final_metrics,
        "future_failure_counts": dict(Counter(failures["diagnosis"])),
        "future_thermodynamic_consistency": consistency_summary,
        "future_impurity_audit": impurity_summary,
        "evidence_boundary": "Literature knowledge and parameter labels are frozen at 2022; current MP phases and EquiformerV3 pretraining are not time frozen.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
