from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from fairchem.core import OCPCalculator
from pymatgen.core import Structure


GPA_A3_TO_EV = 0.006241509
MODEL = "EquiformerV3+DeNS-OAM2"


def parse_side(side: str) -> dict[str, int]:
    species: dict[str, int] = {}
    for token in side.split("+"):
        token = token.strip()
        match = re.match(r"(\d+)\s+(\S+)", token)
        coefficient = int(match.group(1)) if match else 1
        formula = match.group(2) if match else token
        species[formula] = species.get(formula, 0) + coefficient
    return species


def reaction_value(reaction: str, energy_map: dict[str, float]) -> float:
    reactants, products = reaction.split("->")
    left = sum(
        coefficient * energy_map[formula]
        for formula, coefficient in parse_side(reactants).items()
    )
    right = sum(
        coefficient * energy_map[formula]
        for formula, coefficient in parse_side(products).items()
    )
    return right - left


def metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = prediction - reference
    same_sign = np.signbit(reference) == np.signbit(prediction)
    return {
        "n": int(len(reference)),
        "mae_eV": float(np.mean(np.abs(error))),
        "rmse_eV": float(np.sqrt(np.mean(error**2))),
        "bias_eV": float(np.mean(error)),
        "sign_consistency": float(np.mean(same_sign)),
        "max_absolute_error_eV": float(np.max(np.abs(error))),
    }


def formula_from_filename(path: Path) -> str:
    base = path.stem.split("-", 1)[1]
    return base.split("_")[0]


def evaluate_phase_directory(
    calculator: OCPCalculator,
    cif_dir: Path,
    pressure_gpa: float,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) == len(list(cif_dir.glob("*.cif"))):
            return cached

    rows: list[dict[str, object]] = []
    for path in sorted(cif_dir.glob("*.cif")):
        atoms = read(path)
        atoms.calc = calculator
        energy_cell = float(atoms.get_potential_energy())
        volume_a3 = float(atoms.get_volume())
        pv_cell = pressure_gpa * volume_a3 * GPA_A3_TO_EV
        enthalpy_cell = energy_cell + pv_cell
        structure = Structure.from_file(path)
        _, formula_units = structure.composition.get_reduced_composition_and_factor()
        formula_units = float(formula_units)
        rows.append(
            {
                "model": MODEL,
                "pressure_GPa": pressure_gpa,
                "cif_name": path.name,
                "cif_path": str(path),
                "formula_key": formula_from_filename(path),
                "reduced_formula": structure.composition.reduced_formula,
                "formula_units_per_cell": formula_units,
                "energy_cell_eV": energy_cell,
                "volume_cell_A3": volume_a3,
                "pv_cell_eV": pv_cell,
                "enthalpy_cell_eV": enthalpy_cell,
                "energy_per_formula_eV": energy_cell / formula_units,
                "enthalpy_per_formula_eV": enthalpy_cell / formula_units,
            }
        )
        print(f"{pressure_gpa:g} GPa {path.name}", flush=True)
    frame = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def selected_energy_map(frame: pd.DataFrame) -> dict[str, float]:
    # Reproduce the legacy sorted-file traversal: later polymorph files overwrite
    # earlier entries with the same formula key.
    selected = frame.sort_values("cif_name").drop_duplicates("formula_key", keep="last")
    return dict(zip(selected["formula_key"], selected["enthalpy_per_formula_eV"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--output-dir",
        default="/home/zhzhan/synthesis_pathway_screening/results/equiformer_v3_li_ag_au_pressure",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calculator = OCPCalculator(
        checkpoint_path=args.checkpoint,
        cpu=args.device == "cpu",
        seed=0,
    )

    phase_specs = {
        0: Path("/home/zhzhan/LiAuH/0/26 cif"),
        10: Path("/home/zhzhan/LiAuH/10/10GPa_structures"),
    }
    energy_maps: dict[int, dict[str, float]] = {}
    for pressure, cif_dir in phase_specs.items():
        phase_frame = evaluate_phase_directory(
            calculator,
            cif_dir,
            float(pressure),
            output_dir / f"phase_energies_{pressure}GPa.csv",
        )
        energy_maps[pressure] = selected_energy_map(phase_frame)

    case_specs = [
        ("Ag", 0, Path("/home/zhzhan/li/liagh/0/detail.csv")),
        ("Ag", 10, Path("/home/zhzhan/li/liagh/10/detail.csv")),
        ("Au", 0, Path("/home/zhzhan/li/liauh/0/detail.csv")),
        ("Au", 10, Path("/home/zhzhan/li/liauh/10/detail.csv")),
    ]
    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for system, pressure, detail_path in case_specs:
        frame = pd.read_csv(detail_path)
        frame["system"] = system
        frame["pressure_GPa"] = pressure
        frame["EquiformerV3"] = frame["reaction"].map(
            lambda reaction: reaction_value(reaction, energy_maps[pressure])
        )
        frame["Err_EquiformerV3"] = frame["EquiformerV3"] - frame["DFT"]
        frame["SignCorrect_EquiformerV3"] = (
            np.signbit(frame["EquiformerV3"]) == np.signbit(frame["DFT"])
        )
        frame.to_csv(output_dir / f"{system}_{pressure}GPa_reaction_details.csv", index=False)
        detail_frames.append(frame)
        summary_rows.append(
            {
                "system": system,
                "pressure_GPa": pressure,
                **metrics(frame["DFT"].to_numpy(), frame["EquiformerV3"].to_numpy()),
            }
        )

    all_details = pd.concat(detail_frames, ignore_index=True)
    all_details.to_csv(output_dir / "reaction_details_all_68.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "case_summary.csv", index=False)

    pooled = {
        "model": MODEL,
        **metrics(
            all_details["DFT"].to_numpy(),
            all_details["EquiformerV3"].to_numpy(),
        ),
    }
    pd.DataFrame([pooled]).to_csv(output_dir / "pooled_summary.csv", index=False)

    shift_rows: list[dict[str, object]] = []
    shift_detail_frames: list[pd.DataFrame] = []
    for system in ("Ag", "Au"):
        zero = all_details.loc[
            (all_details["system"] == system) & (all_details["pressure_GPa"] == 0),
            ["reaction", "DFT", "EquiformerV3"],
        ].rename(columns={"DFT": "DFT_0", "EquiformerV3": "EquiformerV3_0"})
        ten = all_details.loc[
            (all_details["system"] == system) & (all_details["pressure_GPa"] == 10),
            ["reaction", "DFT", "EquiformerV3"],
        ].rename(columns={"DFT": "DFT_10", "EquiformerV3": "EquiformerV3_10"})
        shift = zero.merge(ten, on="reaction", how="inner")
        shift["system"] = system
        shift["DFT_shift_eV"] = shift["DFT_10"] - shift["DFT_0"]
        shift["EquiformerV3_shift_eV"] = (
            shift["EquiformerV3_10"] - shift["EquiformerV3_0"]
        )
        shift["shift_error_eV"] = shift["EquiformerV3_shift_eV"] - shift["DFT_shift_eV"]
        shift_detail_frames.append(shift)
        shift_rows.append(
            {
                "system": system,
                **metrics(
                    shift["DFT_shift_eV"].to_numpy(),
                    shift["EquiformerV3_shift_eV"].to_numpy(),
                ),
            }
        )
    pd.concat(shift_detail_frames, ignore_index=True).to_csv(
        output_dir / "pressure_shift_details.csv", index=False
    )
    pd.DataFrame(shift_rows).to_csv(output_dir / "pressure_shift_summary.csv", index=False)

    protocol = {
        "model": MODEL,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "pressure_term": "H = E_MLIP + P*V",
        "GPa_A3_to_eV": GPA_A3_TO_EV,
        "polymorph_rule": "sorted CIF traversal; keep last filename for each legacy formula key",
        "reaction_reference": "frozen DFT values in /home/zhzhan/li/*/detail.csv",
        "n_reactions": int(len(all_details)),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pooled": pooled, "cases": summary_rows, "shifts": shift_rows}, indent=2))


if __name__ == "__main__":
    main()
