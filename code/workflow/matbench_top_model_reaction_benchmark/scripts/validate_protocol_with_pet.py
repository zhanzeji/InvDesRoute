from __future__ import annotations

import pandas as pd

from benchmark_common import evaluate_model, write_phase_predictions


SOURCE = "/home/zhzhan/pet-oam-xi/mp_reaction/upet_energies.csv"


def main() -> None:
    source = pd.read_csv(SOURCE)
    rows = []
    for row in source.to_dict("records"):
        rows.append(
            {
                "model": "PET-OAM-XL-protocol-check",
                "formula": row["formula"],
                "cif_name": row["cif_name"],
                "energy_cell_eV": row["E_cell (eV)"],
                "z": row["Z"],
                "energy_per_formula_eV": row["E_per_formula (eV)"],
                "status": "ok",
                "error": "",
            }
        )
    write_phase_predictions("PET-OAM-XL-protocol-check", rows)
    _, _, summary = evaluate_model("PET-OAM-XL-protocol-check")
    print(summary)


if __name__ == "__main__":
    main()

