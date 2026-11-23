#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "multi_route_benchmark"
LITERATURE = PROJECT / "literature_benchmark"
sys.path.insert(0, str(LITERATURE))

from evaluate_route_similarity import (  # noqa: E402
    class_multiset,
    counter_jaccard,
    normalize,
    parse_side,
    route_similarity,
    split_formulas,
)


PROCESS_ELEMENTS = {"H", "C", "N", "O"}
MODERATE = {
    "exact_route_recovered",
    "strong_route_family_match",
    "moderate_route_family_match",
}
STRONG = {"exact_route_recovered", "strong_route_family_match"}


@lru_cache(maxsize=None)
def elements(formula: str) -> frozenset[str]:
    return frozenset(str(element) for element in Composition(formula).elements)


def source_elements(target: str) -> set[str]:
    target_elements = elements(target)
    sources = target_elements - PROCESS_ELEMENTS
    return sources or target_elements


def formula_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@dataclass
class CorpusPrior:
    formula_counts: dict[str, Counter[str]]
    class_counts: dict[str, Counter[str]]
    maxima: dict[str, float]

    @classmethod
    def fit(cls, corpus: pd.DataFrame, excluded_targets: set[str]) -> "CorpusPrior":
        formula_counts: dict[str, Counter[str]] = defaultdict(Counter)
        class_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for _, row in corpus.iterrows():
            try:
                target = normalize(row["target"])
                if target in excluded_targets:
                    continue
                target_sources = source_elements(target)
                precursors = split_formulas(row.get("literature_precursors"))
            except Exception:
                continue
            for formula in precursors:
                represented = elements(formula) & target_sources
                if len(represented) != 1:
                    continue
                element = next(iter(represented))
                if not (elements(formula) - {element}).issubset(PROCESS_ELEMENTS):
                    continue
                formula_counts[element][formula] += 1
                class_counts[element][next(iter(class_multiset({formula}, target)), "other_precursor")] += 1
        maxima = {
            element: math.log1p(max(counts.values())) if counts else 1.0
            for element, counts in formula_counts.items()
        }
        return cls(dict(formula_counts), dict(class_counts), maxima)

    def score(self, target: str, reactants: set[str]) -> float:
        values = []
        for element in sorted(source_elements(target)):
            candidates = [formula for formula in reactants if element in elements(formula)]
            if not candidates:
                values.append(0.0)
                continue
            exact = max(
                math.log1p(self.formula_counts.get(element, Counter()).get(formula, 0))
                / self.maxima.get(element, 1.0)
                for formula in candidates
            )
            class_support = 0.0
            total = sum(self.class_counts.get(element, Counter()).values())
            if total:
                for formula in candidates:
                    formula_class = next(iter(class_multiset({formula}, target)), "other_precursor")
                    class_support = max(
                        class_support,
                        self.class_counts[element].get(formula_class, 0) / total,
                    )
            values.append(0.82 * exact + 0.18 * math.sqrt(class_support))
        coverage = len(set().union(*(elements(formula) for formula in reactants)) & elements(target)) / len(elements(target))
        return float(np.clip(0.90 * np.mean(values) + 0.10 * coverage, 0.0, 1.0))

    def serializable(self, excluded_targets: set[str], source: str) -> dict[str, object]:
        return {
            "source": source,
            "excluded_targets": sorted(excluded_targets),
            "formula_counts": {
                element: dict(sorted(counts.items())) for element, counts in sorted(self.formula_counts.items())
            },
            "class_counts": {
                element: dict(sorted(counts.items())) for element, counts in sorted(self.class_counts.items())
            },
        }


def condition_heavy_score(row: dict[str, object]) -> float:
    weights = {
        "energy_score": 0.10,
        "precursor_prior_score": 0.25,
        "route_template_score": 0.25,
        "practical_filter_score": 0.10,
        "route_complexity_score": 0.07,
        "phase_availability_score": 0.08,
        "condition_simplicity_score": 0.15,
    }
    value = sum(float(row.get(component) or 0.0) * weight for component, weight in weights.items())
    template = float(row.get("route_template_score") or 0.0)
    if template >= 0.95:
        value += 0.06
    elif template >= 0.90:
        value += 0.03
    return min(1.0, value)


def route_redundancy(target: str, left: dict[str, object], right: dict[str, object]) -> float:
    left_reactants = parse_side(left.get("reactants"))
    right_reactants = parse_side(right.get("reactants"))
    exact = formula_jaccard(left_reactants, right_reactants)
    classes = counter_jaccard(
        class_multiset(left_reactants, target),
        class_multiset(right_reactants, target),
    )
    return 0.70 * exact + 0.30 * classes


def rank_routes(
    target: str,
    routes: list[dict[str, object]],
    prior: CorpusPrior,
    prior_weight: float,
    diversity_penalty: float,
    top_k: int,
) -> list[dict[str, object]]:
    candidates = []
    for route in routes:
        route = dict(route)
        reactants = parse_side(route.get("reactants"))
        corpus_score = prior.score(target, reactants)
        base_score = condition_heavy_score(route)
        route["corpus_precursor_prior"] = corpus_score
        route["optimized_rank_score"] = (1.0 - prior_weight) * base_score + prior_weight * corpus_score
        candidates.append(route)
    selected = []
    remaining = candidates
    while remaining and len(selected) < top_k:
        best = max(
            remaining,
            key=lambda route: (
                float(route["optimized_rank_score"])
                - diversity_penalty * max(
                    (route_redundancy(target, route, chosen) for chosen in selected),
                    default=0.0,
                ),
                float(route["optimized_rank_score"]),
                -float(route.get("synth_rank_deltaE_per_target") or 1e12),
                str(route.get("reaction", "")),
            ),
        )
        selected.append(best)
        remaining.remove(best)
    for rank, route in enumerate(selected, start=1):
        route["optimized_rank"] = rank
    return selected


def load_cases(root: Path, targets: set[str]) -> dict[str, list[dict[str, object]]]:
    cases = {}
    for target in sorted(targets):
        path = root / target / "synthesizability_ranked_routes.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        cases[target] = pd.read_csv(path).replace({np.nan: ""}).to_dict("records")
    return cases


def evaluate(
    references: pd.DataFrame,
    cases: dict[str, list[dict[str, object]]],
    prior: CorpusPrior,
    prior_weight: float,
    diversity_penalty: float,
    precursor_only: bool,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    ranked = {
        target: rank_routes(target, routes, prior, prior_weight, diversity_penalty, 5)
        for target, routes in cases.items()
    }
    details = []
    for _, reference in references.iterrows():
        target = normalize(reference["target"])
        literature_precursors = split_formulas(
            reference.get("precursors", reference.get("literature_precursors", ""))
        )
        literature_byproducts = set() if precursor_only else split_formulas(reference.get("literature_byproducts", ""))
        best = None
        strict_exact = False
        for route in ranked[target]:
            candidate = dict(route)
            candidate_precursors = parse_side(candidate.get("reactants"))
            candidate_products = parse_side(candidate.get("products"))
            candidate_byproducts = candidate_products - {target}
            if precursor_only:
                candidate["products"] = f"1*{target}"
                candidate["route_template_reason"] = "external_precursor_prediction"
                strict_exact = strict_exact or candidate_precursors == literature_precursors
            else:
                strict_exact = strict_exact or (
                    candidate_precursors == literature_precursors
                    and candidate_byproducts == literature_byproducts
                )
            similarity = route_similarity(target, literature_precursors, literature_byproducts, candidate)
            if best is None or similarity["route_similarity_score"] > best["route_similarity_score"]:
                best = {**similarity, "matched_reaction": route.get("reaction", ""), "matched_rank": route["optimized_rank"]}
        details.append(
            {
                "route_id": reference.get("route_id", ""),
                "target": target,
                "similarity": float(best["route_similarity_score"]),
                "label": best["route_similarity_label"],
                "moderate_or_better": best["route_similarity_label"] in MODERATE,
                "strong_or_exact": best["route_similarity_label"] in STRONG,
                "strict_exact": strict_exact,
                "matched_rank": best["matched_rank"],
                "matched_reaction": best["matched_reaction"],
            }
        )
    detail = pd.DataFrame(details)
    metrics = {
        "records": len(detail),
        "moderate_or_better": int(detail["moderate_or_better"].sum()),
        "strong_or_exact": int(detail["strong_or_exact"].sum()),
        "strict_exact": int(detail["strict_exact"].sum()),
        "mean_similarity": float(detail["similarity"].mean()),
    }
    selected_rows = []
    for target, routes in ranked.items():
        for route in routes:
            selected_rows.append(
                {
                    "target": target,
                    "rank": route["optimized_rank"],
                    "reaction": route.get("reaction", ""),
                    "reactants": ";".join(sorted(parse_side(route.get("reactants")))),
                    "optimized_rank_score": route["optimized_rank_score"],
                    "corpus_precursor_prior": route["corpus_precursor_prior"],
                    "condition_heavy_score": condition_heavy_score(route),
                }
            )
    return metrics, detail, pd.DataFrame(selected_rows)


def clustered_bootstrap(detail: pd.DataFrame, iterations: int = 10000) -> tuple[float, float]:
    targets = sorted(detail["target"].unique())
    by_target = {target: detail.loc[detail["target"] == target, "moderate_or_better"].mean() for target in targets}
    rng = np.random.default_rng(20260806)
    samples = [
        np.mean([by_target[target] for target in rng.choice(targets, len(targets), replace=True)])
        for _ in range(iterations)
    ]
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "results_equiformer_v3" / "optimized_ranking")
    args = parser.parse_args()

    test_references = pd.read_csv(ROOT / "../generative_model_benchmark/data/benchmark_24x5.csv").fillna("")
    dev_references = pd.read_csv(
        LITERATURE / "results/equiformer_v3_extension53_similarity/route_similarity_eval.csv"
    ).fillna("")
    test_targets = {normalize(value) for value in test_references["target"]}
    dev_references["target"] = dev_references["target"].map(normalize)
    dev_references = dev_references[~dev_references["target"].isin(test_targets)].reset_index(drop=True)
    dev_targets = set(dev_references["target"])
    if test_targets & dev_targets:
        raise RuntimeError(f"development/test target overlap: {sorted(test_targets & dev_targets)}")

    corpus = pd.read_csv(LITERATURE / "ceder_text_mined_routes.csv").fillna("")
    prior = CorpusPrior.fit(corpus, test_targets | dev_targets)
    if len(test_targets & dev_targets):
        raise RuntimeError("development and frozen-test targets must be disjoint")
    dev_cases = load_cases(PROJECT / "results/literature_seeded_extension_equiformer_v3_53", dev_targets)
    test_cases = load_cases(ROOT / "cases_blind_equiformer_v3", test_targets)

    search_rows = []
    for prior_weight, diversity_penalty in itertools.product(
        [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24],
    ):
        metrics, _, _ = evaluate(
            dev_references,
            dev_cases,
            prior,
            prior_weight,
            diversity_penalty,
            precursor_only=False,
        )
        search_rows.append({"prior_weight": prior_weight, "diversity_penalty": diversity_penalty, **metrics})
    search = pd.DataFrame(search_rows).sort_values(
        ["moderate_or_better", "strong_or_exact", "mean_similarity", "prior_weight", "diversity_penalty"],
        ascending=[False, False, False, True, True],
    )
    chosen = search.iloc[0]
    prior_weight = float(chosen["prior_weight"])
    diversity_penalty = float(chosen["diversity_penalty"])

    dev_metrics, dev_detail, dev_ranked = evaluate(
        dev_references, dev_cases, prior, prior_weight, diversity_penalty, precursor_only=False
    )
    test_metrics, test_detail, test_ranked = evaluate(
        test_references, test_cases, prior, prior_weight, diversity_penalty, precursor_only=True
    )
    ci_low, ci_high = clustered_bootstrap(test_detail)
    test_metrics["target_cluster_ci95_low"] = ci_low
    test_metrics["target_cluster_ci95_high"] = ci_high

    ablation_rows = []
    for label, alpha, penalty in [
        ("condition_heavy", 0.0, 0.0),
        ("condition_heavy_plus_corpus", prior_weight, 0.0),
        ("condition_heavy_plus_mmr", 0.0, diversity_penalty),
        ("corpus_informed_mmr", prior_weight, diversity_penalty),
    ]:
        metrics, _, _ = evaluate(
            test_references, test_cases, prior, alpha, penalty, precursor_only=True
        )
        ablation_rows.append(
            {"configuration": label, "prior_weight": alpha, "diversity_penalty": penalty, **metrics}
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    search.to_csv(args.outdir / "development_parameter_search.csv", index=False)
    dev_detail.to_csv(args.outdir / "development_route_detail.csv", index=False)
    dev_ranked.to_csv(args.outdir / "development_top5_routes.csv", index=False)
    test_detail.to_csv(args.outdir / "frozen_test_route_detail.csv", index=False)
    test_ranked.to_csv(args.outdir / "frozen_test_top5_routes.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(args.outdir / "frozen_test_component_ablation.csv", index=False)
    (args.outdir / "frozen_corpus_prior.json").write_text(
        json.dumps(
            prior.serializable(
                test_targets | dev_targets,
                "Kononova et al. text-mined solid-state synthesis corpus; development and frozen-test targets excluded",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    model_predictions = test_ranked[["target", "rank", "reactants", "reaction"]].copy()
    model_predictions.insert(0, "model", "InvDesRoute-corpus-informed")
    model_predictions = model_predictions.rename(columns={"reactants": "precursors", "reaction": "balanced_reaction"})
    model_predictions["byproducts"] = ""
    model_predictions["notes"] = (
        f"target-disjoint development selection; prior_weight={prior_weight}; "
        f"diversity_penalty={diversity_penalty}"
    )
    model_predictions.to_csv(args.outdir / "frozen_test_predictions.csv", index=False)
    summary = {
        "method": "InvDesRoute corpus-informed MMR ranking",
        "development_targets": len(dev_targets),
        "development_records": len(dev_references),
        "test_targets": len(test_targets),
        "test_records": len(test_references),
        "target_overlap": 0,
        "corpus_excluded_targets": len(test_targets | dev_targets),
        "selected_prior_weight": prior_weight,
        "selected_diversity_penalty": diversity_penalty,
        "development_metrics_complete_route": dev_metrics,
        "frozen_test_metrics_precursor_endpoint": test_metrics,
        "selection_rule": "Maximize development top-5 moderate+, then strong/exact, then mean similarity; ties prefer smaller hyperparameters.",
    }
    (args.outdir / "optimized_ranking_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
