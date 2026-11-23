from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CIF_DIR = Path("/home/zhzhan/mattersim-main/hechengyuce/mp_cifs")


def normalize_formula(value: str) -> str:
    return Composition(str(value)).reduced_formula


def structure_records() -> list[dict[str, object]]:
    records = []
    for path in sorted(CIF_DIR.glob("*.cif")):
        structure = Structure.from_file(path)
        reduced, factor = structure.composition.get_reduced_composition_and_factor()
        records.append(
            {
                "cif_path": path,
                "cif_name": path.name,
                "formula": reduced.reduced_formula,
                "z": float(factor),
                "n_sites": int(structure.num_sites),
            }
        )
    return records


def write_phase_predictions(model: str, rows: list[dict[str, object]]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{model}_phase_energies.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def parse_side(side: str) -> dict[str, int]:
    species: dict[str, int] = {}
    for token in side.split("+"):
        token = token.strip()
        match = re.match(r"(\d+)\s+(\S+)", token)
        coefficient = int(match.group(1)) if match else 1
        formula = normalize_formula(match.group(2) if match else token)
        species[formula] = species.get(formula, 0) + coefficient
    return species


def reaction_energy(reaction: str, energy_map: dict[str, float]) -> float:
    left, right = reaction.split("->")
    reactants = parse_side(left)
    products = parse_side(right)
    return sum(value * energy_map[key] for key, value in products.items()) - sum(
        value * energy_map[key] for key, value in reactants.items()
    )


def load_dft_reference() -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    frame = pd.read_csv(DATA / "mp_dft_phase_energies.csv")
    formula_energy: dict[str, float] = {}
    formula_epa: dict[str, float] = {}
    formula_atoms: dict[str, int] = {}
    for row in frame.to_dict("records"):
        if str(row["energy_per_atom (eV)"]) == "FAILED":
            continue
        formula = normalize_formula(row["formula"])
        atom_count = int(Composition(str(row["formula"])).num_atoms)
        energy_per_atom = float(row["energy_per_atom (eV)"])
        formula_energy[formula] = energy_per_atom * atom_count
        formula_epa[formula] = energy_per_atom
        formula_atoms[formula] = atom_count
    return formula_energy, formula_epa, formula_atoms


def evaluate_model(model: str) -> tuple[Path, Path, dict[str, object]]:
    phase_path = RESULTS / f"{model}_phase_energies.csv"
    phase_frame = pd.read_csv(phase_path)
    valid = phase_frame.loc[phase_frame["status"] == "ok"].copy()
    valid["formula"] = valid["formula"].map(normalize_formula)
    valid = valid.drop_duplicates("formula", keep="last")
    model_energy = dict(zip(valid["formula"], valid["energy_per_formula_eV"]))
    dft_energy, dft_epa, atom_counts = load_dft_reference()

    phase_rows = []
    for formula in sorted(model_energy.keys() & dft_energy.keys()):
        predicted_epa = float(model_energy[formula]) / atom_counts[formula]
        error = predicted_epa - dft_epa[formula]
        phase_rows.append(
            {
                "model": model,
                "formula": formula,
                "predicted_energy_per_atom_eV": predicted_epa,
                "mp_dft_energy_per_atom_eV": dft_epa[formula],
                "error_eV_per_atom": error,
                "absolute_error_eV_per_atom": abs(error),
            }
        )

    reaction_rows = []
    reactions = pd.read_csv(DATA / "reactions_416.csv")
    for reaction in reactions["reaction"]:
        try:
            predicted = reaction_energy(reaction, model_energy)
            reference = reaction_energy(reaction, dft_energy)
        except KeyError:
            continue
        error = predicted - reference
        reaction_rows.append(
            {
                "model": model,
                "reaction": reaction,
                "predicted_delta_e_eV": predicted,
                "mp_dft_delta_e_eV": reference,
                "error_eV": error,
                "absolute_error_eV": abs(error),
                "same_sign": bool(predicted * reference > 0),
            }
        )

    phase_out = RESULTS / f"{model}_phase_comparison.csv"
    reaction_out = RESULTS / f"{model}_reaction_comparison.csv"
    pd.DataFrame(phase_rows).to_csv(phase_out, index=False)
    pd.DataFrame(reaction_rows).to_csv(reaction_out, index=False)

    phase_errors = np.array([row["error_eV_per_atom"] for row in phase_rows], dtype=float)
    reaction_errors = np.array([row["error_eV"] for row in reaction_rows], dtype=float)
    same_sign = np.array([row["same_sign"] for row in reaction_rows], dtype=float)
    summary = {
        "model": model,
        "n_phase": len(phase_rows),
        "phase_mae_eV_per_atom": float(np.mean(np.abs(phase_errors))),
        "phase_rmse_eV_per_atom": float(np.sqrt(np.mean(phase_errors**2))),
        "phase_bias_eV_per_atom": float(np.mean(phase_errors)),
        "n_reaction": len(reaction_rows),
        "reaction_mae_eV": float(np.mean(np.abs(reaction_errors))),
        "reaction_rmse_eV": float(np.sqrt(np.mean(reaction_errors**2))),
        "reaction_bias_eV": float(np.mean(reaction_errors)),
        "reaction_sign_consistency": float(np.mean(same_sign)),
    }
    summary_path = RESULTS / f"{model}_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return phase_out, reaction_out, summary

