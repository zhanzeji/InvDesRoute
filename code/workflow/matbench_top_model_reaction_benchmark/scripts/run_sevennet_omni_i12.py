from __future__ import annotations

import argparse

from ase.io import read
from sevenn.calculator import SevenNetCalculator

from benchmark_common import evaluate_model, structure_records, write_phase_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model_name = "SevenNet-Omni-i12"
    calculator = SevenNetCalculator(
        model=args.checkpoint,
        device=args.device,
        modal="mpa",
    )
    rows = []
    for record in structure_records():
        try:
            atoms = read(record["cif_path"])
            atoms.calc = calculator
            energy_cell = float(atoms.get_potential_energy())
            rows.append(
                {
                    "model": model_name,
                    "formula": record["formula"],
                    "cif_name": record["cif_name"],
                    "energy_cell_eV": energy_cell,
                    "z": record["z"],
                    "energy_per_formula_eV": energy_cell / float(record["z"]),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "model": model_name,
                    "formula": record["formula"],
                    "cif_name": record["cif_name"],
                    "energy_cell_eV": "",
                    "z": record["z"],
                    "energy_per_formula_eV": "",
                    "status": "failed",
                    "error": repr(error),
                }
            )
        print(record["cif_name"], rows[-1]["status"], flush=True)
    write_phase_predictions(model_name, rows)
    _, _, summary = evaluate_model(model_name)
    print(summary)


if __name__ == "__main__":
    main()

