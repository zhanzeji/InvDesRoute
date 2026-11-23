from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from pymatgen.core import Composition, Element


ELEMENTS = tuple(Element.from_Z(index).symbol for index in range(1, 119))
ELEMENT_INDEX = {element: index for index, element in enumerate(ELEMENTS)}
VOLATILE = {"H", "C", "N", "O", "F", "Cl", "Br", "I"}


@lru_cache(maxsize=None)
def normalize(value: str) -> str:
    return Composition(str(value).strip().split("·", 1)[0]).reduced_formula


def split_formulas(value: str) -> tuple[str, ...]:
    return tuple(sorted({normalize(token) for token in str(value or "").split(";") if token.strip()}))


@lru_cache(maxsize=None)
def composition_vector(formula: str) -> np.ndarray:
    composition = Composition(normalize(formula))
    vector = np.zeros(len(ELEMENTS) * 2 + 4, dtype=np.float32)
    total = float(composition.num_atoms)
    for element, amount in composition.items():
        index = ELEMENT_INDEX[str(element)]
        vector[index] = float(amount) / total
        vector[len(ELEMENTS) + index] = 1.0
    vector[-4] = len(composition.elements) / 10.0
    vector[-3] = math.log1p(total) / 5.0
    vector[-2] = float(any(str(element) == "O" for element in composition.elements))
    vector[-1] = float(any(str(element) in {"H", "C", "N"} for element in composition.elements))
    return vector


def source_elements(target: str) -> tuple[str, ...]:
    elements = [str(element) for element in Composition(normalize(target)).elements]
    sources = [element for element in elements if element not in VOLATILE]
    return tuple(sources or elements)


def single_source_precursor(precursor: str, target: str) -> str | None:
    target_sources = set(source_elements(target))
    overlap = target_sources & {str(element) for element in Composition(precursor).elements}
    return next(iter(overlap)) if len(overlap) == 1 else None


def precursor_template(precursor: str, source: str) -> str:
    composition = Composition(normalize(precursor))
    parts = []
    for element, amount in composition.get_el_amt_dict().items():
        symbol = "M" if element == source else element
        parts.append(f"{symbol}:{float(amount):g}")
    return "|".join(parts)


def instantiate_template(template: str, source: str) -> str | None:
    try:
        amounts = {}
        for token in template.split("|"):
            element, amount = token.split(":", 1)
            amounts[source if element == "M" else element] = float(amount)
        return Composition(amounts).reduced_formula
    except Exception:
        return None


def route_feature(precursors: tuple[str, ...]) -> np.ndarray:
    vectors = np.stack([composition_vector(formula) for formula in precursors])
    feature = vectors.mean(axis=0)
    return np.concatenate([feature, np.array([len(precursors) / 8.0], dtype=np.float32)])
