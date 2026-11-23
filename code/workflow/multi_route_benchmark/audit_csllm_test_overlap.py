#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
LITERATURE_DIR = ROOT.parent / "literature_benchmark"
sys.path.insert(0, str(LITERATURE_DIR))

from evaluate_route_similarity import normalize, split_formulas  # noqa: E402


def target_from_prompt(prompt: str) -> str:
    match = re.search(r'"([^"%]+)%?"', prompt)
    if not match:
        return ""
    try:
        return normalize(match.group(1))
    except Exception:
        return ""


def precursor_tuple(output: str) -> tuple[str, ...]:
    try:
        values = ast.literal_eval(output)
    except Exception:
        return ()
    if not isinstance(values, (list, tuple)):
        return ()
    formulas = set()
    for value in values:
        try:
            formulas.add(normalize(str(value).split("·", 1)[0]))
        except Exception:
            continue
    return tuple(sorted(formulas))


def main() -> None:
    source = ROOT / "external_baselines" / "CSLLM" / "data" / "precursors" / "test" / "test_llm_precursors.json"
    literature = pd.read_csv(ROOT / "multi_route_literature_dataset.csv").fillna("")
    targets = sorted({normalize(value) for value in literature["target"]})
    reference_sets = {
        target: {
            tuple(sorted(split_formulas(row["literature_precursors"])))
            for _, row in literature[literature["target"].map(normalize) == target].iterrows()
        }
        for target in targets
    }

    payload = json.loads(source.read_text(encoding="utf-8"))
    records = []
    for instance in payload["instances"]:
        target = target_from_prompt(instance.get("input", ""))
        if target not in reference_sets:
            continue
        precursors = precursor_tuple(instance.get("output", ""))
        records.append(
            {
                "target": target,
                "official_test_input": instance.get("input", ""),
                "official_test_output": instance.get("output", ""),
                "normalized_precursors": ";".join(precursors),
                "exact_selected_route_overlap": precursors in reference_sets[target],
            }
        )

    detail = pd.DataFrame(records)
    per_target = []
    for target in targets:
        group = detail[detail["target"] == target]
        per_target.append(
            {
                "target": target,
                "present_in_official_test": not group.empty,
                "official_test_records": len(group),
                "official_test_unique_precursor_sets": group["normalized_precursors"].nunique(),
                "records_overlapping_selected_routes": int(group["exact_selected_route_overlap"].sum()),
                "selected_benchmark_routes": len(reference_sets[target]),
            }
        )
    target_table = pd.DataFrame(per_target)
    outdir = ROOT / "results" / "external_models"
    outdir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(outdir / "csllm_official_test_overlap_records.csv", index=False)
    target_table.to_csv(outdir / "csllm_official_test_overlap_by_target.csv", index=False)

    summary = {
        "official_test_instances": len(payload["instances"]),
        "benchmark_targets": len(targets),
        "benchmark_targets_present_in_official_test": int(target_table["present_in_official_test"].sum()),
        "matching_official_test_records": len(detail),
        "unique_target_precursor_pairs": int(detail[["target", "normalized_precursors"]].drop_duplicates().shape[0]),
        "records_exactly_overlapping_selected_benchmark_routes": int(detail["exact_selected_route_overlap"].sum()),
        "training_split_manifest_available": False,
        "target_disjointness_from_training_confirmed": False,
        "interpretation": "All benchmark targets occur in the released CSLLM precursor test file. The released repository does not include the corresponding precursor training manifest, so target-level disjointness from training cannot be independently confirmed.",
    }
    (outdir / "csllm_official_test_overlap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
