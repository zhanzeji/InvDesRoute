#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition


PROJECT = Path(__file__).resolve().parents[2]


def run_target(target: str, output_root: Path, python: str, force: bool) -> dict[str, object]:
    target_dir = output_root / target
    required_outputs = [
        target_dir / "formation_reactions.csv",
        target_dir / "synthesizability_ranked_routes.csv",
        target_dir / "run_metadata.json",
    ]
    if all(path.exists() for path in required_outputs) and not force:
        return {"target": target, "status": "cached", "returncode": 0}
    targeted_max_reactants = min(4, max(2, len(Composition(target).elements)))
    command = [
        python,
        str(PROJECT / "run_screening.py"),
        "--target-formula",
        target,
        "--database",
        "mp",
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
        str(targeted_max_reactants),
        "--targeted-max-byproducts",
        "3",
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
    process = subprocess.run(command, text=True, capture_output=True, env=os.environ.copy())
    (target_dir / "workflow_stdout.log").write_text(process.stdout, encoding="utf-8")
    (target_dir / "workflow_stderr.log").write_text(process.stderr, encoding="utf-8")
    return {
        "target": target,
        "status": "ok" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "last_stdout_line": process.stdout.strip().splitlines()[-1] if process.stdout.strip() else "",
        "last_stderr_line": process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blind generic-library workflows for post-2022 targets.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default="/home/zhzhan/miniconda3/envs/mattersim/bin/python")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not (os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")):
        raise SystemExit("Set MP_API_KEY or PMG_MAPI_KEY in the environment")
    targets = sorted(pd.read_csv(args.manifest)["target"].astype(str).unique())
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_target, target, args.output_root, args.python, args.force): target
            for target in targets
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            rows.append(result)
            print(f"[{index}/{len(targets)}] {result['target']}: {result['status']}", flush=True)
    pd.DataFrame(rows).sort_values("target").to_csv(args.output_root / "workflow_manifest.csv", index=False)
    failures = [row for row in rows if row["status"] == "failed"]
    if failures:
        raise SystemExit(f"{len(failures)} targets failed")


if __name__ == "__main__":
    main()
