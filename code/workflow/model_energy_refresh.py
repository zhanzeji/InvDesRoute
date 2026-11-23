#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from pymatgen.core import Composition, Structure


PROJECT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT / "results" / "mp_pet_benchmark_120"
DEFAULT_OUTPUT = PROJECT / "results" / "grace_benchmark_120"
DEFAULT_CHECKPOINT = (
    PROJECT
    / "matbench_top_model_reaction_benchmark"
    / "models"
    / "grace_savedmodel"
    / "GRACE-3L-OMAT-large-ft-AM"
)
DEFAULT_EQUIFORMER_CHECKPOINT = (
    PROJECT / "leading_mlips" / "checkpoints" / "omat24-mptrj-salex_gradient.pt"
)

MODEL_SPECS = {
    "grace": {
        "model_name": "GRACE-3L-OAM-L",
        "column_prefix": "GRACE-3L-OAM-L",
        "phase_filename": "phase_pool_grace.csv",
        "default_output": PROJECT / "results" / "grace_benchmark_120",
        "default_checkpoint": DEFAULT_CHECKPOINT,
    },
    "equiformer-v3": {
        "model_name": "EquiformerV3+DeNS-OAM2",
        "column_prefix": "EquiformerV3+DeNS-OAM2",
        "phase_filename": "phase_pool_equiformer_v3.csv",
        "default_output": PROJECT / "results" / "equiformer_v3_benchmark_120",
        "default_checkpoint": DEFAULT_EQUIFORMER_CHECKPOINT,
    },
}


def normalize_formula(value: str) -> str:
    return Composition(str(value)).reduced_formula


def collect_structures(source_root: Path) -> pd.DataFrame:
    rows: dict[str, dict[str, object]] = {}
    for phase_path in sorted(source_root.glob("*/phase_pool_raw.csv")):
        target = phase_path.parent.name
        frame = pd.read_csv(phase_path)
        for row in frame.to_dict("records"):
            cif_value = row.get("cif_path")
            if not isinstance(cif_value, str) or not cif_value or not Path(cif_value).exists():
                continue
            cif_path = str(Path(cif_value).resolve())
            if cif_path not in rows:
                structure = Structure.from_file(cif_path)
                _, factor = structure.composition.get_reduced_composition_and_factor()
                rows[cif_path] = {
                    "cif_path": cif_path,
                    "formula": normalize_formula(row["formula"]),
                    "source_id": row.get("source_id", ""),
                    "z": float(factor),
                    "n_sites": int(structure.num_sites),
                    "first_target": target,
                }
    return pd.DataFrame(rows.values()).sort_values(["formula", "cif_path"]).reset_index(drop=True)


def grace_calculator(checkpoint: Path):
    import tensorflow as tf
    from tensorpotential.calculator import TPCalculator

    model = tf.saved_model.load(str(checkpoint))
    signature = model.signatures["compute"].structured_input_signature[1]
    model.compute_specs = dict(signature)
    return TPCalculator(
        model=model,
        cutoff=6.0,
        pad_neighbors_fraction=0.05,
        pad_atoms_number=10,
    )


def equiformer_calculator(checkpoint: Path, device: str):
    from fairchem.core import OCPCalculator

    return OCPCalculator(checkpoint_path=str(checkpoint), cpu=device == "cpu", seed=0)


def evaluate_model(
    source_root: Path,
    checkpoint: Path,
    output_csv: Path,
    limit: int | None,
    model_key: str,
    device: str,
    shard_index: int,
    num_shards: int,
) -> pd.DataFrame:
    records = collect_structures(source_root)
    records = records.iloc[shard_index::num_shards].copy()
    if limit is not None:
        records = records.head(limit).copy()
    calculator = (
        grace_calculator(checkpoint)
        if model_key == "grace"
        else equiformer_calculator(checkpoint, device)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(output_csv) if output_csv.exists() and output_csv.stat().st_size else pd.DataFrame()
    completed = set(existing.loc[existing.get("status", "") == "ok", "cif_path"]) if len(existing) else set()
    rows = existing.to_dict("records") if len(existing) else []
    for index, record in enumerate(records.to_dict("records"), start=1):
        if record["cif_path"] in completed:
            continue
        result = dict(record)
        try:
            atoms = read(record["cif_path"])
            atoms.calc = calculator
            energy_cell = float(atoms.get_potential_energy())
            result.update(
                {
                    "energy_cell_eV": energy_cell,
                    "energy_per_formula_eV": energy_cell / float(record["z"]),
                    "energy_per_atom_eV": energy_cell / float(record["n_sites"]),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as error:
            result.update(
                {
                    "energy_cell_eV": np.nan,
                    "energy_per_formula_eV": np.nan,
                    "energy_per_atom_eV": np.nan,
                    "status": "failed",
                    "error": repr(error),
                }
            )
        rows.append(result)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"[{index}/{len(records)}] {record['formula']} {result['status']}", flush=True)
    return pd.DataFrame(rows)


def parse_species(value: object) -> dict[str, float]:
    species: dict[str, float] = {}
    if not isinstance(value, str):
        return species
    for token in value.split("|"):
        token = token.strip()
        if not token:
            continue
        coefficient_text, formula_text = token.split("*", 1)
        formula = normalize_formula(formula_text.strip())
        species[formula] = species.get(formula, 0.0) + float(coefficient_text)
    return species


def reaction_energy(row: pd.Series, energy_map: dict[str, float]) -> float:
    reactants = parse_species(row.get("reactants"))
    products = parse_species(row.get("products"))
    if not reactants or not products:
        return math.nan
    formulas = set(reactants) | set(products)
    if any(formula not in energy_map for formula in formulas):
        return math.nan
    return sum(coefficient * energy_map[formula] for formula, coefficient in products.items()) - sum(
        coefficient * energy_map[formula] for formula, coefficient in reactants.items()
    )


def top_overlap(frame: pd.DataFrame, left: str, right: str, n: int = 5) -> int:
    valid = frame.dropna(subset=[left, right])
    if valid.empty:
        return 0
    left_routes = set(valid.nsmallest(min(n, len(valid)), left)["reaction"])
    right_routes = set(valid.nsmallest(min(n, len(valid)), right)["reaction"])
    return len(left_routes & right_routes)


def refresh_networks(
    source_root: Path,
    output_root: Path,
    phase_predictions: pd.DataFrame,
    model_spec: dict[str, object],
) -> pd.DataFrame:
    if not output_root.exists():
        shutil.copytree(source_root, output_root)
    energy_by_path = {
        str(Path(row["cif_path"]).resolve()): float(row["energy_per_formula_eV"])
        for row in phase_predictions.loc[phase_predictions["status"] == "ok"].to_dict("records")
    }
    summaries = []
    for target_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        phase_path = target_dir / "phase_pool_raw.csv"
        if not phase_path.exists():
            continue
        phases = pd.read_csv(phase_path)
        energy_formula_col = f"{model_spec['column_prefix']}_energy_per_formula"
        energy_atom_col = f"{model_spec['column_prefix']}_energy_per_atom"
        phases[energy_formula_col] = phases["cif_path"].map(
            lambda value: energy_by_path.get(str(Path(value).resolve())) if isinstance(value, str) and value else np.nan
        )
        phases[energy_atom_col] = phases[energy_formula_col] / phases["n_atoms"]
        phases.to_csv(target_dir / str(model_spec["phase_filename"]), index=False)
        energy_map = dict(zip(phases["formula"].map(normalize_formula), phases[energy_formula_col]))
        target_summary: dict[str, object] = {
            "target": normalize_formula(target_dir.name),
            "n_phases": int(len(phases)),
            "n_phases_with_model": int(phases[energy_formula_col].notna().sum()),
        }
        for filename in ("formation_reactions.csv", "decomposition_reactions.csv"):
            reaction_path = target_dir / filename
            if not reaction_path.exists() or not reaction_path.stat().st_size:
                continue
            reactions = pd.read_csv(reaction_path)
            values = reactions.apply(lambda row: reaction_energy(row, energy_map), axis=1)
            model_total_col = f"DeltaE_{model_spec['column_prefix']}"
            model_col = f"{model_total_col}_per_target"
            reactions[model_total_col] = values
            reactions[model_col] = values / reactions["target_coefficient"]
            reactions.to_csv(target_dir / filename, index=False)
            if filename != "formation_reactions.csv":
                continue
            ref_col = "DeltaE_MP_FORM_per_target"
            valid = reactions.dropna(subset=[model_col, ref_col]).copy()
            errors = valid[model_col] - valid[ref_col]
            practical = reactions.loc[reactions.get("practical_route", False) == True].dropna(subset=[model_col])
            target_summary.update(
                {
                    "n_formation_reactions": int(len(reactions)),
                    "n_comparable_reactions": int(len(valid)),
                    "reaction_mae_eV_per_target": float(errors.abs().mean()) if len(valid) else np.nan,
                    "reaction_rmse_eV_per_target": float(np.sqrt((errors**2).mean())) if len(valid) else np.nan,
                    "reaction_bias_eV_per_target": float(errors.mean()) if len(valid) else np.nan,
                    "sign_consistency": float(((valid[model_col] * valid[ref_col]) > 0).mean()) if len(valid) else np.nan,
                    "top1_agreement": bool(
                        len(valid)
                        and valid.nsmallest(1, model_col).iloc[0]["reaction"]
                        == valid.nsmallest(1, ref_col).iloc[0]["reaction"]
                    ),
                    "top5_overlap": top_overlap(valid, model_col, ref_col),
                    "has_exergonic_practical_route": bool(len(practical) and practical[model_col].min() < -0.05),
                    "best_practical_reaction": practical.nsmallest(1, model_col).iloc[0]["reaction"] if len(practical) else "",
                    "best_practical_deltaE_eV_per_target": float(practical[model_col].min()) if len(practical) else np.nan,
                }
            )
        stability_path = target_dir / "phase_stability.csv"
        if stability_path.exists() and stability_path.stat().st_size:
            stability = pd.read_csv(stability_path)
            target_rows = stability.loc[stability["formula"].map(normalize_formula) == normalize_formula(target_dir.name)]
            target_summary["target_ehull_eV_per_atom"] = (
                float(target_rows.iloc[0]["energy_above_hull"]) if len(target_rows) else np.nan
            )
        summaries.append(target_summary)
    summary = pd.DataFrame(summaries).sort_values("target").reset_index(drop=True)
    summary.to_csv(output_root / "model_workflow_target_summary.csv", index=False)
    aggregate = {
        "model": model_spec["model_name"],
        "n_targets": int(len(summary)),
        "n_phases": int(summary["n_phases"].sum()),
        "n_phases_with_model": int(summary["n_phases_with_model"].sum()),
        "n_formation_reactions": int(summary["n_formation_reactions"].sum()),
        "n_comparable_reactions": int(summary["n_comparable_reactions"].sum()),
        "mean_target_mae_eV_per_target": float(summary["reaction_mae_eV_per_target"].mean()),
        "median_target_mae_eV_per_target": float(summary["reaction_mae_eV_per_target"].median()),
        "mean_sign_consistency": float(summary["sign_consistency"].mean()),
        "top1_agreement_count": int(summary["top1_agreement"].sum()),
        "mean_top5_overlap": float(summary["top5_overlap"].mean()),
        "exergonic_practical_count": int(summary["has_exergonic_practical_route"].sum()),
        "exergonic_practical_low_ehull_count": int(
            (summary["has_exergonic_practical_route"] & (summary["target_ehull_eV_per_atom"] <= 0.10)).sum()
        ),
    }
    (output_root / "model_workflow_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="grace")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--phase-output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--apply-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= index < --num-shards")
    model_spec = MODEL_SPECS[args.model]
    output_root = args.output_root or Path(model_spec["default_output"])
    checkpoint = args.checkpoint or Path(model_spec["default_checkpoint"])
    phase_output = args.phase_output or output_root.parent / f"{args.model}_workflow_phase_energies.csv"
    if args.apply_only:
        predictions = pd.read_csv(phase_output)
    else:
        predictions = evaluate_model(
            args.source_root,
            checkpoint,
            phase_output,
            args.limit,
            args.model,
            args.device,
            args.shard_index,
            args.num_shards,
        )
    if args.limit is None and args.num_shards == 1:
        refresh_networks(args.source_root, output_root, predictions, model_spec)


if __name__ == "__main__":
    main()
