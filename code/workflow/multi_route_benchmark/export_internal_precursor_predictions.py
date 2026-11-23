#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
LITERATURE_DIR = ROOT.parent / "literature_benchmark"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LITERATURE_DIR))

from evaluate_multi_route_benchmark import (  # noqa: E402
    build_leave_target_out_frequency,
    parse_side,
    rank_routes,
)
from evaluate_route_similarity import normalize  # noqa: E402


MODES = {
    "Production-energy-network": "pet_energy",
    "Practical-production-network": "practical_pet_energy",
    "Chemistry-prior-network": "chemistry_prior_generator",
    "InvDesRoute": "synth_score",
    "InvDesRoute-diverse": "diverse_synth_score",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-root', type=Path, default=ROOT / 'cases_blind')
    parser.add_argument('--output', type=Path, default=ROOT / 'external_model_interface' / 'internal_network_precursor_top20.csv')
    args = parser.parse_args()
    dataset = pd.read_csv(ROOT / "multi_route_literature_dataset.csv").fillna("")
    frequencies = build_leave_target_out_frequency(dataset, LITERATURE_DIR / "ceder_text_mined_routes.csv")
    rows = []
    for target in sorted(dataset["target"].map(normalize).unique()):
        routes = pd.read_csv(args.case_root / target / "synthesizability_ranked_routes.csv").fillna("").to_dict("records")
        for route in routes:
            classes = Counter(str(route.get("reactant_classes", "")).split(";"))
            route["leave_target_out_class_frequency"] = frequencies.get(
                ";".join(f"{name}:{count}" for name, count in sorted(classes.items()) if name), 0
            )
        for model, mode in MODES.items():
            ranked = rank_routes([dict(route) for route in routes], mode)
            seen = set()
            output_rank = 0
            for route in ranked:
                precursors = tuple(sorted(parse_side(route.get("reactants"))))
                if not precursors or precursors in seen:
                    continue
                seen.add(precursors)
                output_rank += 1
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "rank": output_rank,
                        "precursors": ";".join(precursors),
                        "byproducts": "",
                        "balanced_reaction": route.get("reaction", ""),
                        "notes": f"source route rank={route.get('evaluation_rank', '')}",
                    }
                )
                if output_rank >= 20:
                    break
        ranked = sorted(
            [dict(route) for route in routes],
            key=lambda route: (-condition_heavy_score(route), str(route.get("reaction", ""))),
        )
        seen = set()
        output_rank = 0
        for route in ranked:
            precursors = tuple(sorted(parse_side(route.get("reactants"))))
            if not precursors or precursors in seen:
                continue
            seen.add(precursors)
            output_rank += 1
            rows.append(
                {
                    "model": "InvDesRoute-condition-heavy",
                    "target": target,
                    "rank": output_rank,
                    "precursors": ";".join(precursors),
                    "byproducts": "",
                    "balanced_reaction": route.get("reaction", ""),
                    "notes": "candidate score profile selected by target-stratified validation",
                }
            )
            if output_rank >= 20:
                break
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
