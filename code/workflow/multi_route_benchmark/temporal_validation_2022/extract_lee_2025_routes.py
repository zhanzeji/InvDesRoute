#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition


DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.I)


def normalize_formula(value: object) -> str:
    return Composition(str(value).strip()).reduced_formula


def normalized_map(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for formula, coefficient in value.items():
        try:
            normalized = normalize_formula(formula)
            amount = float(coefficient)
        except Exception:
            return {}
        if amount <= 0:
            return {}
        result[normalized] = result.get(normalized, 0.0) + amount
    return result


def clean_target(record: dict[str, object]) -> tuple[str, str, float | None] | None:
    targets = record.get("target")
    if not isinstance(targets, list) or len(targets) != 1 or not isinstance(targets[0], dict):
        return None
    target = targets[0]
    if target.get("additives") or target.get("amounts_vars") or target.get("elements_vars"):
        return None
    formula = target.get("material_formula")
    try:
        normalized = normalize_formula(formula)
    except Exception:
        return None
    if not normalized or len(Composition(normalized).elements) < 2:
        return None
    mp_id = str(target.get("mp_id") or "").strip()
    ehull = target.get("mp_eabovehull")
    try:
        ehull = float(ehull) if ehull is not None else None
    except Exception:
        ehull = None
    return normalized, mp_id, ehull


def extract_target_reaction(record: dict[str, object], target: str) -> tuple[dict[str, float], dict[str, float], str] | None:
    reactions = record.get("target_reaction")
    if not isinstance(reactions, list):
        return None
    for entry in reactions:
        if not isinstance(entry, list) or len(entry) < 2 or not isinstance(entry[1], dict):
            continue
        left = normalized_map(entry[1].get("left"))
        right = normalized_map(entry[1].get("right"))
        if not left or not right or target not in right:
            continue
        expression = str(entry[3] if len(entry) > 3 else "").strip()
        return left, right, expression
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract auditable route records from the Lee et al. 80,806-recipe dataset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with gzip.open(args.input, "rt", encoding="utf-8") as handle:
        records = json.load(handle)

    rows = []
    rejection_counts: dict[str, int] = {}
    for source_index, record in enumerate(records):
        if not isinstance(record, dict):
            rejection_counts["not_dictionary"] = rejection_counts.get("not_dictionary", 0) + 1
            continue
        target_data = clean_target(record)
        if target_data is None:
            rejection_counts["invalid_or_nonstoichiometric_target"] = rejection_counts.get("invalid_or_nonstoichiometric_target", 0) + 1
            continue
        target, mp_id, ehull = target_data
        doi = str(record.get("DOI") or "").strip().lower()
        if not DOI_PATTERN.match(doi):
            rejection_counts["invalid_doi"] = rejection_counts.get("invalid_doi", 0) + 1
            continue
        reaction = extract_target_reaction(record, target)
        if reaction is None:
            rejection_counts["missing_balanced_target_reaction"] = rejection_counts.get("missing_balanced_target_reaction", 0) + 1
            continue
        left, right, expression = reaction
        byproducts = {formula: amount for formula, amount in right.items() if formula != target}
        impurity_formulas = []
        for impurity in record.get("impurity_phase") or []:
            if not isinstance(impurity, dict):
                continue
            try:
                impurity_formulas.append(normalize_formula(impurity.get("material_formula")))
            except Exception:
                continue
        rows.append(
            {
                "source_dataset": "Lee et al. 2025 solid-state recipes with impurity phases",
                "source_index": source_index,
                "doi": doi,
                "target": target,
                "target_mp_id": mp_id,
                "target_ehull_eV_per_atom": ehull,
                "literature_precursors": ";".join(sorted(left)),
                "literature_byproducts": ";".join(sorted(byproducts)),
                "literature_route_text": expression,
                "impurity_phases": ";".join(sorted(set(impurity_formulas))),
                "has_reported_impurity": bool(impurity_formulas),
                "n_precursors": len(left),
                "n_byproducts": len(byproducts),
            }
        )

    frame = pd.DataFrame(rows)
    before = len(frame)
    frame = frame.drop_duplicates(
        ["doi", "target", "literature_precursors", "literature_byproducts", "literature_route_text"]
    ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = {
        "source_records": len(records),
        "auditable_route_records_before_deduplication": before,
        "auditable_route_records": len(frame),
        "unique_targets": int(frame["target"].nunique()),
        "unique_dois": int(frame["doi"].nunique()),
        "reported_impurity_routes": int(frame["has_reported_impurity"].sum()),
        "rejection_counts": rejection_counts,
        "source_sha256": "59fb0608e594721a4055fb6a623f180651f2d7aef737a5d4e13db63595b039c0",
    }
    (args.output.parent / "extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
