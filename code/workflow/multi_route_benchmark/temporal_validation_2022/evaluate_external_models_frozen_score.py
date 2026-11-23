#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
BENCHMARK = PROJECT / "multi_route_benchmark"
GENERATION = PROJECT / "generative_model_benchmark"
LITERATURE = PROJECT / "literature_benchmark"
HERE = Path(__file__).resolve().parent
SPLIT = HERE / "split_2022_energy_complete"
RESULTS = HERE / "results_2022" / "external_models_frozen_score"

sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LITERATURE))

from evaluate_route_similarity import normalize, parse_side, route_similarity, split_formulas  # noqa: E402
from evaluate_2022_temporal_benchmark import (  # noqa: E402
    MODERATE_LABELS,
    PROFILES,
    STRONG_LABELS,
    prepare_features,
    rank_routes,
)
from optimize_ranking_target_disjoint import CorpusPrior  # noqa: E402


PROFILE = "condition_heavy"
ALPHA = 0.30
LAMBDA = 0.10


def load_cases(targets: set[str]) -> dict[str, list[dict[str, object]]]:
    cases: dict[str, list[dict[str, object]]] = {}
    root = BENCHMARK / "cases_blind_equiformer_v3"
    for target in sorted(targets):
        path = root / target / "synthesizability_ranked_routes.csv"
        cases[target] = pd.read_csv(path).replace({np.nan: ""}).to_dict("records")
    return cases


def evaluate_references(
    references: pd.DataFrame,
    cases: dict[str, list[dict[str, object]]],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    selected = {
        target: rank_routes(target, routes, PROFILE, ALPHA, LAMBDA, 5)
        for target, routes in cases.items()
    }
    detail_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for target, routes in selected.items():
        for rank, route in enumerate(routes, start=1):
            prediction_rows.append(
                {
                    "model": "InvDesRoute-corpus-informed",
                    "target": target,
                    "rank": rank,
                    "precursors": ";".join(sorted(parse_side(route.get("reactants")))),
                    "balanced_reaction": route.get("reaction", ""),
                    "byproducts": "",
                    "score": route.get("evaluation_rank_score", ""),
                    "base_score": route.get("evaluation_base_score", ""),
                    "historical_corpus_score": route.get("historical_corpus_score", ""),
                    "notes": (
                        f"frozen 2022 policy; profile={PROFILE}; alpha={ALPHA:.2f}; "
                        f"lambda={LAMBDA:.2f}"
                    ),
                }
            )

    for _, reference in references.iterrows():
        target = normalize(reference["target"])
        literature_precursors = split_formulas(reference["precursors"])
        best = None
        strict_exact = False
        for rank, route in enumerate(selected[target], start=1):
            candidate = dict(route)
            candidate_precursors = parse_side(candidate.get("reactants"))
            strict_exact = strict_exact or candidate_precursors == literature_precursors
            candidate["products"] = f"1*{target}"
            candidate["route_template_reason"] = "external_precursor_prediction"
            similarity = route_similarity(target, literature_precursors, set(), candidate)
            key = (float(similarity["route_similarity_score"]), -rank)
            if best is None or key > best[0]:
                best = (key, rank, route, similarity)
        assert best is not None
        _, rank, route, similarity = best
        detail_rows.append(
            {
                "route_id": reference["route_id"],
                "target": target,
                "doi": reference["doi"],
                "strict_exact": strict_exact,
                "strong_or_exact": similarity["route_similarity_label"] in STRONG_LABELS,
                "moderate_or_better": similarity["route_similarity_label"] in MODERATE_LABELS,
                "precursor_recall": float(similarity["exact_reactant_recall"]),
                "route_similarity": float(similarity["route_similarity_score"]),
                "matched_rank": rank,
                "matched_reaction": route.get("reaction", ""),
            }
        )
    detail = pd.DataFrame(detail_rows)
    metrics = {
        "records": len(detail),
        "targets": int(detail["target"].nunique()),
        "strict_exact": int(detail["strict_exact"].sum()),
        "strong_or_exact": int(detail["strong_or_exact"].sum()),
        "moderate_or_better": int(detail["moderate_or_better"].sum()),
        "mean_precursor_recall": float(detail["precursor_recall"].mean()),
        "mean_route_similarity": float(detail["route_similarity"].mean()),
    }
    return metrics, detail, pd.DataFrame(prediction_rows)


def main() -> None:
    references = pd.read_csv(GENERATION / "data" / "benchmark_24x5.csv").fillna("")
    references["target"] = references["target"].map(normalize)
    targets = set(references["target"])

    historical = pd.read_csv(
        SPLIT / "historical_prior_routes_through_2020.csv", low_memory=False
    ).fillna("")
    historical["target"] = historical["target"].map(normalize)
    historical = historical[~historical["target"].isin(targets)].copy()
    prior = CorpusPrior.fit(historical, targets)
    cases = prepare_features(load_cases(targets), prior)
    metrics, detail, predictions = evaluate_references(references, cases)

    RESULTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(RESULTS / "invdesroute_frozen_2022_detail.csv", index=False)
    predictions.to_csv(RESULTS / "invdesroute_frozen_2022_predictions.csv", index=False)
    summary = {
        "policy_source": "2021-2022 development split; no 24-target external-test tuning",
        "profile": PROFILE,
        "weights": dict(zip(
            (
                "energy_score",
                "precursor_prior_score",
                "route_template_score",
                "practical_filter_score",
                "route_complexity_score",
                "phase_availability_score",
                "condition_simplicity_score",
            ),
            PROFILES[PROFILE],
        )),
        "historical_prior_cutoff": 2020,
        "historical_prior_routes_after_test_target_exclusion": len(historical),
        "alpha": ALPHA,
        "lambda": LAMBDA,
        "metrics": metrics,
    }
    (RESULTS / "invdesroute_frozen_2022_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
