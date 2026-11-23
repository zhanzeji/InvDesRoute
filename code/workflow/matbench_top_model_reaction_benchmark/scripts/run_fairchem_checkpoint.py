from __future__ import annotations

import argparse

from ase.io import read
from fairchem.core import OCPCalculator

from benchmark_common import evaluate_model, structure_records, write_phase_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    calculator = OCPCalculator(
        checkpoint_path=args.checkpoint,
        cpu=args.device == "cpu",
        seed=0,
    )
    records = structure_records()
    if args.limit is not None:
        records = records[: args.limit]

    rows = []
    for record in records:
        try:
            atoms = read(record["cif_path"])
            atoms.calc = calculator
            energy_cell = float(atoms.get_potential_energy())
            rows.append(
                {
                    "model": args.model_name,
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
                    "model": args.model_name,
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

    write_phase_predictions(args.model_name, rows)
    if args.limit is None:
        _, _, summary = evaluate_model(args.model_name)
        print(summary)


if __name__ == "__main__":
    main()
