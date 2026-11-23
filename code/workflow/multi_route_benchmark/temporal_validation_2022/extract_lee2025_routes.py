#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition


def normalize(value: object) -> str:
    return Composition(str(value).strip()).reduced_formula


def safe_normalize(value: object) -> str:
    try:
        return normalize(value)
    except Exception:
        return ""


def material_formula(material: object) -> str:
    if not isinstance(material, dict):
        return ""
    value = material.get("material_formula") or material.get("material_string")
    return safe_normalize(value)


def side_formulas(side: object) -> tuple[dict[str, float], str]:
    if not isinstance(side, dict):
        return {}, "invalid_side"
    formulas: dict[str, float] = {}
    for formula, coefficient in side.items():
        normalized = safe_normalize(formula)
        if not normalized:
            return {}, f"invalid_formula:{formula}"
        try:
            value = float(coefficient)
        except Exception:
            return {}, f"invalid_coefficient:{coefficient}"
        if not value > 0:
            return {}, f"nonpositive_coefficient:{coefficient}"
        formulas[normalized] = formulas.get(normalized, 0.0) + value
    return formulas, ""


def balanced(left: dict[str, float], right: dict[str, float], tolerance: float = 0.02) -> bool:
    elements = sorted(
        {str(element) for formula in [*left, *right] for element in Composition(formula).elements}
    )
    for element in elements:
        lhs = sum(coefficient * Composition(formula)[element] for formula, coefficient in left.items())
        rhs = sum(coefficient * Composition(formula)[element] for formula, coefficient in right.items())
        scale = max(1.0, abs(lhs), abs(rhs))
        if abs(lhs - rhs) > tolerance * scale:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract independently validated routes from the Lee et al. 2025 dataset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with gzip.open(args.input, "rt", encoding="utf-8") as handle:
        records = json.load(handle)

    rows = []
    for record_index, record in enumerate(records):
        targets = record.get("target") or []
        target_material = targets[0] if targets else {}
        target = material_formula(target_material)
        target_mp_id = target_material.get("mp_id", "") if isinstance(target_material, dict) else ""
        target_ehull = target_material.get("mp_eabovehull", "") if isinstance(target_material, dict) else ""
        variable_target = bool(
            isinstance(target_material, dict)
            and (
                target_material.get("amounts_vars")
                or target_material.get("elements_vars")
                or target_material.get("oxygen_deficiency") not in (None, "", 0, 0.0)
                or target_material.get("additives")
            )
        )
        reactions = record.get("target_reaction") or []
        if not reactions:
            rows.append(
                {
                    "source_record_id": f"lee2025_{record_index:06d}",
                    "doi": str(record.get("DOI") or "").lower(),
                    "target": target,
                    "target_mp_id": target_mp_id,
                    "target_ehull_eV_per_atom": target_ehull,
                    "phase_pure_report": not bool(record.get("impurity_phase")),
                    "route_valid": False,
                    "exclusion_reason": "missing_balanced_target_reaction",
                }
            )
            continue
        for reaction_index, reaction in enumerate(reactions):
            route_id = f"lee2025_{record_index:06d}_r{reaction_index:02d}"
            reaction_target = safe_normalize(reaction[0]) if isinstance(reaction, list) and reaction else ""
            expression = reaction[1] if isinstance(reaction, list) and len(reaction) > 1 else {}
            left, left_error = side_formulas(expression.get("left") if isinstance(expression, dict) else None)
            right, right_error = side_formulas(expression.get("right") if isinstance(expression, dict) else None)
            target_formula = target or reaction_target
            reasons = []
            if not target_formula:
                reasons.append("invalid_target_formula")
            if variable_target:
                reasons.append("variable_or_doped_target")
            if left_error:
                reasons.append(left_error)
            if right_error:
                reasons.append(right_error)
            if target_formula and target_formula not in right:
                reasons.append("target_not_on_product_side")
            if left and right and not balanced(left, right):
                reasons.append("independent_balance_check_failed")
            target_elements = {str(element) for element in Composition(target_formula).elements} if target_formula else set()
            left_elements = {
                str(element) for formula in left for element in Composition(formula).elements
            }
            if target_elements and not target_elements.issubset(left_elements):
                reasons.append("target_element_coverage_failed")
            impurity_formulas = sorted(
                {
                    material_formula(material)
                    for material in record.get("impurity_phase") or []
                    if material_formula(material)
                }
            )
            rows.append(
                {
                    "route_id": route_id,
                    "source_record_id": f"lee2025_{record_index:06d}",
                    "doi": str(record.get("DOI") or "").strip().lower(),
                    "target": target_formula,
                    "target_mp_id": target_mp_id,
                    "target_ehull_eV_per_atom": target_ehull,
                    "literature_precursors": ";".join(sorted(left)),
                    "literature_byproducts": ";".join(sorted(set(right) - {target_formula})),
                    "literature_route_text": reaction[3] if isinstance(reaction, list) and len(reaction) > 3 else "",
                    "phase_pure_report": not bool(record.get("impurity_phase")),
                    "reported_impurity_phases": ";".join(impurity_formulas),
                    "n_precursors": len(left),
                    "n_byproducts": len(set(right) - {target_formula}),
                    "n_target_elements": len(target_elements),
                    "route_valid": not reasons,
                    "exclusion_reason": ";".join(reasons),
                }
            )

    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "lee2025_all_extracted_routes.csv", index=False)
    valid = frame.loc[frame["route_valid"].eq(True)].copy()
    valid.to_csv(args.output_dir / "lee2025_valid_balanced_routes.csv", index=False)
    summary = {
        "source_records": len(records),
        "extracted_route_rows": len(frame),
        "valid_balanced_routes": len(valid),
        "valid_unique_dois": int(valid["doi"].nunique()),
        "valid_unique_targets": int(valid["target"].nunique()),
        "valid_phase_pure_routes": int(valid["phase_pure_report"].sum()),
        "valid_impurity_reported_routes": int((~valid["phase_pure_report"]).sum()),
        "exclusion_reasons": frame.loc[~frame["route_valid"], "exclusion_reason"].value_counts().to_dict(),
    }
    (args.output_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
