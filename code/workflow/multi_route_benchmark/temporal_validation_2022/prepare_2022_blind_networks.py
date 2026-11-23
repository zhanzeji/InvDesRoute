#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build generic-library route networks for the 2022 temporal benchmark.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default="/home/zhzhan/miniconda3/envs/mattersim/bin/python")
    parser.add_argument("--split", default="", help="Optional manifest split to process")
    parser.add_argument("--run-name", default="all", help="Suffix for shard-specific manifests")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY is required")
    manifest = pd.read_csv(args.manifest)
    if args.split:
        manifest = manifest.loc[manifest["split"].eq(args.split)].copy()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row in manifest.sort_values(["split", "target"]).reset_index(drop=True).iterrows():
        target = row["target"]
        target_dir = args.output_root / target
        output = target_dir / "synthesizability_ranked_routes.csv"
        status = "reused"
        if args.force or not output.exists():
            command = [
                args.python,
                str(PROJECT / "run_screening.py"),
                "--target-formula",
                target,
                "--database",
                "mp",
                "--mp-api-key",
                api_key,
                "--mp-max-phases",
                "160",
                "--include-common-precursors",
                "--include-gas-reservoirs",
                "--include-virtual-common-precursors",
                "--max-reactants",
                "2",
                "--max-byproducts",
                "1",
                "--byproducts-reservoir-only",
                "--max-products",
                "2",
                "--max-total-coeff",
                "100",
                "--include-targeted-precursor-routes",
                "--targeted-max-reactants",
                "3",
                "--targeted-max-byproducts",
                "4",
                "--targeted-max-total-coeff",
                "180",
                "--practical-max-atoms",
                "24",
                "--practical-max-coeff-sum-per-target",
                "100",
                "--output-dir",
                str(target_dir),
            ]
            target_dir.mkdir(parents=True, exist_ok=True)
            process = subprocess.run(command, text=True, capture_output=True)
            (target_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
            (target_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
            status = "ok" if process.returncode == 0 else "failed"
        n_phases = n_routes = 0
        if (target_dir / "phase_stability.csv").exists():
            n_phases = len(pd.read_csv(target_dir / "phase_stability.csv"))
        if output.exists():
            n_routes = len(pd.read_csv(output))
        rows.append(
            {
                "target": target,
                "split": row["split"],
                "target_mp_id": row["target_mp_id"],
                "status": status,
                "n_phases": n_phases,
                "n_routes": n_routes,
                "literature_precursors_injected": False,
            }
        )
        pd.DataFrame(rows).to_csv(args.output_root / f"network_manifest_{args.run_name}.csv", index=False)
        print(f"[{index + 1}/{len(manifest)}] {target}: {status}, {n_phases} phases, {n_routes} routes", flush=True)

    summary = {
        "targets": len(rows),
        "successful": sum(row["status"] in {"ok", "reused"} and row["n_routes"] > 0 for row in rows),
        "failed": [row["target"] for row in rows if row["status"] == "failed" or row["n_routes"] == 0],
        "candidate_policy": "Materials Project phase systems plus fixed generic precursor/reservoir library; no literature-route labels.",
    }
    (args.output_root / f"network_summary_{args.run_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
