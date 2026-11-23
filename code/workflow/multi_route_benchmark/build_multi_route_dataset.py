#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pymatgen.core import Composition


PROJECT = Path(__file__).resolve().parents[1]
LITERATURE_DIR = PROJECT / "literature_benchmark"
sys.path.insert(0, str(LITERATURE_DIR))

from evaluate_route_similarity import precursor_class  # noqa: E402
from select_ceder_validation_subset import (  # noqa: E402
    is_auditable_route,
    normalize,
    source_doi,
    split_formulas,
)


TARGETS = {
    "perovskite_and_ferroelectric": ["BaTiO3", "SrTiO3", "CaTiO3", "BaZrO3", "LaAlO3", "LaFeO3"],
    "battery_and_ion_storage": ["LiMn2O4", "LiCoO2", "LiFePO4", "Li2MnO3", "LiMnO2", "Li4Ti5O12", "Li7La3Zr2O12"],
    "silicate_phosphate_and_titanate": ["Li4SiO4", "Li2TiO3", "Mg2SiO4", "Ca3(PO4)2"],
    "functional_complex_oxide": ["Y3Al5O12", "MgAl2O4", "BaFe12O19", "LiNbO3", "NaNbO3"],
    "non_oxide": ["Ti3AlC2", "Bi2Te3"],
}

KNOWN_EXTRACTION_ARTIFACTS = {
    "AlO",
    "LaCO3",
    "Li2O3",
    "LiCO3",
    "Mn2(CO3)7",
    "NaCO3",
    "SiO4",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_list(value: str | None) -> list[str]:
    return sorted(set(split_formulas(value)))


def route_pair_key(row: dict[str, str]) -> tuple[str, str]:
    return row["literature_precursors"], row["literature_byproducts"]


def class_signature(target: str, formulas: list[str]) -> str:
    classes = Counter(precursor_class(formula, target) for formula in formulas)
    return ";".join(f"{name}:{count}" for name, count in sorted(classes.items()))


def condition_signature(text: str | None) -> str:
    text = str(text or "")
    temperatures = re.findall(r"(?:temperature|values)[^0-9-]*([0-9]{2,4}(?:\.[0-9]+)?)", text, flags=re.I)
    atmospheres = re.findall(r"(?:atmosphere|gas)[^A-Za-z]*(air|oxygen|O2|nitrogen|N2|argon|Ar|vacuum)", text, flags=re.I)
    operations = [
        name
        for name in ["MixingOperation", "HeatingOperation", "DryingOperation", "GrindingOperation", "QuenchingOperation"]
        if name in text
    ]
    return "|".join([",".join(temperatures[:3]), ",".join(sorted(set(atmospheres))), ",".join(operations)])


def canonicalize_row(row: dict[str, str], target: str, category: str) -> dict[str, str]:
    precursors = normalized_list(row.get("literature_precursors"))
    byproducts = normalized_list(row.get("literature_byproducts"))
    selected = dict(row)
    selected["target"] = target
    selected["material_category"] = category
    selected["literature_precursors"] = ";".join(precursors)
    selected["literature_byproducts"] = ";".join(byproducts)
    selected["doi"] = source_doi(row.get("source_note"))
    selected["precursor_class_signature"] = class_signature(target, precursors)
    selected["byproduct_class_signature"] = class_signature(target, byproducts)
    selected["condition_signature"] = condition_signature(row.get("condition_summary"))
    selected["route_pair_signature"] = " -> ".join([selected["literature_precursors"], selected["literature_byproducts"]])
    return selected


def has_standalone_target_product(route_text: str | None, target: str) -> bool:
    """Reject solid solutions or composites that were reduced to a pure target label."""
    product_side = str(route_text or "").split("==", 1)[-1]
    target_pattern = re.escape(target)
    return bool(re.search(rf"(?:^|\+)\s*[0-9.]*\s*{target_pattern}\s*(?:\+|$)", product_side))


def contains_known_extraction_artifact(precursors: list[str], byproducts: list[str]) -> bool:
    return bool((set(precursors) | set(byproducts)) & KNOWN_EXTRACTION_ARTIFACTS)


def candidate_sort_key(row: dict[str, str]) -> tuple[int, int, str]:
    condition_richness = int(bool(row["condition_signature"].strip("|")))
    formula_count = len(row["literature_precursors"].split(";")) + len(row["literature_byproducts"].split(";"))
    return (-condition_richness, formula_count, row["doi"])


def select_diverse_routes(rows: list[dict[str, str]], routes_per_target: int) -> list[dict[str, str]]:
    unique_records: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["doi"], row["literature_route_text"])
        unique_records.setdefault(key, row)
    remaining = sorted(unique_records.values(), key=candidate_sort_key)
    selected: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_classes: set[tuple[str, str]] = set()
    seen_dois: set[str] = set()

    while remaining and len(selected) < routes_per_target:
        best_index = 0
        best_key = None
        for index, row in enumerate(remaining):
            pair = route_pair_key(row)
            classes = (row["precursor_class_signature"], row["byproduct_class_signature"])
            novelty = (
                int(pair not in seen_pairs),
                int(classes not in seen_classes),
                int(row["doi"] not in seen_dois),
                int(bool(row["condition_signature"].strip("|"))),
                -len(row["literature_precursors"].split(";")),
            )
            if best_key is None or novelty > best_key:
                best_key = novelty
                best_index = index
        row = remaining.pop(best_index)
        pair = route_pair_key(row)
        classes = (row["precursor_class_signature"], row["byproduct_class_signature"])
        row = dict(row)
        row["selection_tier"] = "new_precursor_pair" if pair not in seen_pairs else "independent_doi_or_condition"
        selected.append(row)
        seen_pairs.add(pair)
        seen_classes.add(classes)
        seen_dois.add(row["doi"])
    return selected


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def main() -> None:
    input_path = LITERATURE_DIR / "ceder_text_mined_routes.csv"
    output_dir = PROJECT / "multi_route_benchmark"
    target_to_category = {
        normalize(target): category
        for category, targets in TARGETS.items()
        for target in targets
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(input_path):
        try:
            target = normalize(row["target"])
        except Exception:
            continue
        if target not in target_to_category:
            continue
        precursors = split_formulas(row.get("literature_precursors"))
        byproducts = split_formulas(row.get("literature_byproducts"))
        if not is_auditable_route(row, target, precursors, byproducts):
            continue
        if not has_standalone_target_product(row.get("literature_route_text"), target):
            continue
        if contains_known_extraction_artifact(precursors, byproducts):
            continue
        if len(set(precursors)) > 3:
            continue
        if re.search(r"(?:^|[=+])\s*-\d", str(row.get("literature_route_text", ""))):
            continue
        grouped[target].append(canonicalize_row(row, target, target_to_category[target]))

    dataset_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for target, category in target_to_category.items():
        selected = select_diverse_routes(grouped[target], routes_per_target=5)
        if len(selected) != 5:
            raise RuntimeError(f"{target}: expected 5 auditable routes, found {len(selected)}")
        unique_pairs = len({route_pair_key(row) for row in selected})
        unique_classes = len({(row["precursor_class_signature"], row["byproduct_class_signature"]) for row in selected})
        for route_index, row in enumerate(selected, start=1):
            row = dict(row)
            row["route_id"] = f"{target}__R{route_index:02d}"
            row["route_index_within_target"] = route_index
            dataset_rows.append(row)
        target_rows.append(
            {
                "target": target,
                "material_category": category,
                "n_selected_routes": len(selected),
                "n_unique_precursor_byproduct_pairs": unique_pairs,
                "n_unique_class_signatures": unique_classes,
                "n_available_auditable_records": len(grouped[target]),
                "n_available_dois": len({row["doi"] for row in grouped[target]}),
            }
        )

    route_fields = [
        "route_id",
        "target",
        "material_category",
        "route_index_within_target",
        "literature_precursors",
        "literature_byproducts",
        "literature_route_text",
        "precursor_class_signature",
        "byproduct_class_signature",
        "route_pair_signature",
        "selection_tier",
        "doi",
        "synthesis_type",
        "condition_signature",
        "condition_summary",
        "source_title",
        "source_note",
    ]
    target_fields = [
        "target",
        "material_category",
        "n_selected_routes",
        "n_unique_precursor_byproduct_pairs",
        "n_unique_class_signatures",
        "n_available_auditable_records",
        "n_available_dois",
    ]
    write_csv(output_dir / "multi_route_literature_dataset.csv", dataset_rows, route_fields)
    write_csv(output_dir / "multi_route_target_manifest.csv", target_rows, target_fields)
    print({"targets": len(target_rows), "routes": len(dataset_rows), "categories": len(TARGETS)})
    print(output_dir / "multi_route_literature_dataset.csv")


if __name__ == "__main__":
    main()
