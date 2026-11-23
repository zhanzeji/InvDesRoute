#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine the frozen 120- and 53-target audits.")
    parser.add_argument("--shared-120", type=Path, required=True)
    parser.add_argument("--extension-53", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shared = json.loads(args.shared_120.read_text(encoding="utf-8"))
    extension = json.loads(args.extension_53.read_text(encoding="utf-8"))
    base = shared["EquiformerV3"]
    n_base = int(shared["n_targets"])
    n_extension = int(extension["n_targets_comparable"])
    total = n_base + n_extension
    output = {
        "model": "EquiformerV3+DeNS-OAM2",
        "definition": "Frozen same-route audit: 17,085 shared 120-target routes plus 2,769 extension routes.",
        "n_targets": total,
        "n_comparable_routes": int(shared["n_shared_reactions"]) + int(extension["n_comparable_routes"]),
        "mean_target_mae_eV_per_target": (
            n_base * float(base["macro_target_mae_eV_per_target"])
            + n_extension * float(extension["mean_target_mae_eV_per_target"])
        ) / total,
        "median_target_mae_note": "Not pooled from aggregate medians; see target-level source tables.",
        "mean_target_sign_consistency": (
            n_base * float(base["macro_sign_consistency"])
            + n_extension * float(extension["mean_target_sign_consistency"])
        ) / total,
        "top1_agreement_count": int(base["top1_agreement_count"])
        + int(extension["top1_agreement_count"]),
        "mean_top5_overlap": (
            n_base * float(base["mean_top5_overlap"])
            + n_extension * float(extension["mean_top5_overlap"])
        ) / total,
        "source_files": [str(args.shared_120), str(args.extension_53)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
