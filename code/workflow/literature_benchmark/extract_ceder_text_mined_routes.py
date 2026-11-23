#!/usr/bin/env python3
"""Extract target/precursor routes from Ceder text-mined synthesis JSON.

The 2020 public release stores records under a top-level ``reactions`` key with
explicit ``target``, ``precursors`` and balanced ``reaction_string`` fields.  We
prefer those fields and only fall back to a recursive formula search for older
or partially incompatible schemas.  The output follows the same columns as the
curated seed literature route tables, so it can be used by the same validation
scripts after a downstream quality audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import re
from pathlib import Path
from typing import Any

from pymatgen.core import Composition


FORMULA_LIKE = re.compile(r"^[A-Z][A-Za-z0-9().]*$")
DEFAULT_INPUT = Path("/home/zhzhan/synthesis_pathway_screening/literature_benchmark/raw/solid-state_dataset_20200713.json.xz")
DEFAULT_OUTPUT = Path("/home/zhzhan/synthesis_pathway_screening/literature_benchmark/ceder_text_mined_routes.csv")


def formula_like(value: str) -> bool:
    value = value.strip()
    if not value or len(value) > 80:
        return False
    if not FORMULA_LIKE.match(value):
        return False
    return bool(re.search(r"[A-Z]", value))


def normalize_formula(value: str) -> str:
    return Composition(str(value).strip()).reduced_formula


def safe_formula(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if not formula_like(value):
        return ""
    try:
        return normalize_formula(value)
    except Exception:
        return ""


def material_formula(material: Any) -> str:
    if isinstance(material, str):
        return safe_formula(material)
    if not isinstance(material, dict):
        return ""
    for key in ["material_formula", "material_string", "formula"]:
        value = safe_formula(material.get(key))
        if value:
            return value
    composition = material.get("composition")
    if isinstance(composition, list) and composition:
        value = safe_formula(composition[0].get("formula") if isinstance(composition[0], dict) else "")
        if value:
            return value
    return ""


def reaction_side_formulas(record: dict[str, Any], side: str) -> list[str]:
    reaction = record.get("reaction")
    if not isinstance(reaction, dict):
        return []
    values = []
    for item in reaction.get(side, []) or []:
        if isinstance(item, dict):
            formula = material_formula(item.get("material"))
            if formula:
                values.append(formula)
    return values


def extract_schema_record(record: dict[str, Any]) -> tuple[str, list[str], list[str], str, str, str]:
    """Return target, precursors, byproducts, route text, conditions, DOI."""
    target = material_formula(record.get("target"))
    if not target:
        targets = record.get("targets_string") or []
        if isinstance(targets, list) and targets:
            target = safe_formula(targets[0])
    precursors = []
    for item in record.get("precursors") or []:
        formula = material_formula(item)
        if formula and formula != target:
            precursors.append(formula)
    for formula in reaction_side_formulas(record, "left_side"):
        if formula and formula != target:
            precursors.append(formula)
    right_side = reaction_side_formulas(record, "right_side")
    byproducts = [formula for formula in right_side if formula != target]
    route_text = str(record.get("reaction_string") or "").strip()
    if not route_text and precursors and target:
        route_text = f"{' + '.join(precursors)} -> {target}"

    condition_parts = []
    for operation in record.get("operations") or []:
        if not isinstance(operation, dict):
            continue
        op_type = operation.get("type")
        if op_type:
            condition_parts.append(str(op_type))
        conditions = operation.get("conditions")
        if isinstance(conditions, dict):
            for key, value in conditions.items():
                if value not in (None, "", [], {}):
                    condition_parts.append(f"{key}={value}")
    return (
        target,
        sorted(set(precursors)),
        sorted(set(byproducts)),
        route_text,
        "; ".join(condition_parts),
        str(record.get("doi") or "").strip(),
    )


def walk_values(obj: Any, key_hint: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(walk_values(value, str(key).lower()))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(walk_values(value, key_hint))
    elif isinstance(obj, str):
        clean = obj.strip()
        if formula_like(clean):
            found.append((key_hint, clean))
    return found


def extract_record(record: dict[str, Any]) -> tuple[str, list[str], str]:
    target, precursors, _byproducts, route_text, _condition, _doi = extract_schema_record(record)
    if target and precursors:
        return target, precursors, route_text
    values = walk_values(record)
    target_candidates = []
    precursor_candidates = []
    for key, value in values:
        if any(token in key for token in ["target", "product", "material"]):
            target_candidates.append(value)
        if any(token in key for token in ["precursor", "reactant", "starting"]):
            precursor_candidates.append(value)
    if not target_candidates:
        return "", [], ""
    target = target_candidates[0]
    precursors = sorted({item for item in precursor_candidates if item != target})
    route_text = f"{' + '.join(precursors)} -> {target}" if precursors else f"unknown -> {target}"
    return target, precursors, route_text


def load_json_xz(path: Path) -> Any:
    with lzma.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def iter_records(data: Any):
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        for key in ["reactions", "data", "entries", "documents"]:
            if isinstance(data.get(key), list):
                yield from data[key]
                return
        yield data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    data = load_json_xz(args.input)
    rows = []
    for index, record in enumerate(iter_records(data)):
        if args.max_records and index >= args.max_records:
            break
        if not isinstance(record, dict):
            continue
        target, precursors, byproducts, route_text, condition_summary, doi = extract_schema_record(record)
        if not target or not precursors:
            target, precursors, route_text = extract_record(record)
            byproducts = []
            condition_summary = ""
            doi = ""
        if not target or not precursors:
            continue
        rows.append(
            {
                "target": target,
                "literature_precursors": ";".join(precursors),
                "literature_byproducts": ";".join(byproducts),
                "literature_route_text": route_text,
                "synthesis_type": str(record.get("synthesis_type") or "text-mined solid-state"),
                "condition_summary": condition_summary,
                "source_title": "CederGroupHub text-mined-synthesis_public solid-state dataset",
                "source_note": f"Automatically extracted from public text-mined synthesis JSON; DOI={doi}; requires downstream schema/quality audit.",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "target",
            "literature_precursors",
            "literature_byproducts",
            "literature_route_text",
            "synthesis_type",
            "condition_summary",
            "source_title",
            "source_note",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} routes to {args.output}")


if __name__ == "__main__":
    main()
