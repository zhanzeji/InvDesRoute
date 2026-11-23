#!/usr/bin/env python3
from __future__ import annotations

import itertools
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition


ROOT = Path(__file__).resolve().parent
LITERATURE_DIR = ROOT.parent / "literature_benchmark"
sys.path.insert(0, str(LITERATURE_DIR))

from evaluate_route_similarity import normalize, split_formulas  # noqa: E402


PROCESS_ELEMENTS = {"H", "C", "N", "O"}
KNOWN_ARTIFACTS = {"AlO", "LaCO3", "Li2O3", "LiCO3", "Mn2(CO3)7", "NaCO3", "SiO4"}


def elements(formula: str) -> set[str]:
    return {str(element) for element in Composition(formula).elements}


def composition_vector(formula: str, element_order: list[str]) -> list[float]:
    values = Composition(formula).fractional_composition.as_dict()
    return [float(values.get(element, 0.0)) for element in element_order]


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    return numerator / (norm_left * norm_right) if norm_left and norm_right else 0.0


def valid_precursors(target: str, precursors: set[str]) -> bool:
    if not precursors or precursors & KNOWN_ARTIFACTS:
        return False
    target_elements = elements(target)
    precursor_elements = set().union(*(elements(formula) for formula in precursors))
    return target_elements.issubset(precursor_elements) and (precursor_elements - target_elements).issubset(PROCESS_ELEMENTS)


def build_training_corpus(corpus: pd.DataFrame, held_out: set[str]) -> list[dict[str, object]]:
    rows = []
    for _, row in corpus.iterrows():
        try:
            target = normalize(row["target"])
        except Exception:
            continue
        if target in held_out:
            continue
        precursors = split_formulas(row.get("literature_precursors"))
        if not precursors or precursors & KNOWN_ARTIFACTS:
            continue
        try:
            precursor_elements = set().union(*(elements(formula) for formula in precursors))
        except Exception:
            continue
        if not elements(target).issubset(precursor_elements):
            continue
        rows.append({"target": target, "precursors": frozenset(precursors)})
    return rows


def elementwise_frequency_predictions(target: str, training: list[dict[str, object]], top_k: int) -> list[tuple[tuple[str, ...], float]]:
    target_elements = elements(target)
    source_elements = sorted(target_elements - {"O"}) or sorted(target_elements)
    frequencies: dict[str, Counter[str]] = {element: Counter() for element in source_elements}
    for row in training:
        for formula in row["precursors"]:
            formula_elements = elements(formula)
            represented = formula_elements & set(source_elements)
            if len(represented) != 1:
                continue
            element = next(iter(represented))
            if (formula_elements - {element}).issubset(PROCESS_ELEMENTS):
                frequencies[element][formula] += 1
    options = []
    for element in source_elements:
        ranked = frequencies[element].most_common(8)
        if element not in {formula for formula, _ in ranked}:
            ranked.append((element, 0.25))
        options.append(ranked)
    candidates = []
    for combination in itertools.product(*options):
        formulas = tuple(sorted(set(formula for formula, _ in combination)))
        if not valid_precursors(target, set(formulas)):
            continue
        score = sum(math.log1p(float(count)) for _, count in combination)
        candidates.append((formulas, score))
    best_by_set = {}
    for formulas, score in candidates:
        best_by_set[formulas] = max(score, best_by_set.get(formulas, -math.inf))
    return sorted(best_by_set.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def composition_retrieval_predictions(target: str, training: list[dict[str, object]], top_k: int) -> list[tuple[tuple[str, ...], float]]:
    target_elements = elements(target)
    element_order = sorted(target_elements)
    target_vector = composition_vector(target, element_order)
    best_by_set = {}
    for row in training:
        precursors = set(row["precursors"])
        if not valid_precursors(target, precursors):
            continue
        training_elements = elements(row["target"])
        element_jaccard = len(target_elements & training_elements) / len(target_elements | training_elements)
        similarity = 0.75 * cosine(target_vector, composition_vector(row["target"], element_order)) + 0.25 * element_jaccard
        formulas = tuple(sorted(precursors))
        best_by_set[formulas] = max(similarity, best_by_set.get(formulas, -math.inf))
    return sorted(best_by_set.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def main() -> None:
    dataset = pd.read_csv(ROOT / "multi_route_literature_dataset.csv").fillna("")
    targets = sorted(dataset["target"].map(normalize).unique())
    held_out = set(targets)
    corpus = pd.read_csv(LITERATURE_DIR / "ceder_text_mined_routes.csv").fillna("")
    training = build_training_corpus(corpus, held_out)
    output_rows = []
    diagnostics = []
    for target in targets:
        predictions = {
            "Elementwise-frequency": elementwise_frequency_predictions(target, training, 20),
            "Composition-retrieval": composition_retrieval_predictions(target, training, 20),
        }
        for model, candidates in predictions.items():
            diagnostics.append({"model": model, "target": target, "n_predictions": len(candidates)})
            for rank, (formulas, score) in enumerate(candidates, start=1):
                output_rows.append(
                    {
                        "model": model,
                        "target": target,
                        "rank": rank,
                        "precursors": ";".join(formulas),
                        "byproducts": "",
                        "balanced_reaction": "",
                        "notes": f"leakage-controlled corpus score={score:.8f}",
                    }
                )
        print(target, {model: len(values) for model, values in predictions.items()})
    outdir = ROOT / "external_model_interface"
    pd.DataFrame(output_rows).to_csv(outdir / "corpus_generation_baselines_top20.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(outdir / "corpus_generation_baselines_diagnostics.csv", index=False)
    (outdir / "corpus_generation_training_manifest.txt").write_text(
        f"source_rows={len(corpus)}\nusable_training_rows={len(training)}\nheld_out_targets={len(held_out)}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
