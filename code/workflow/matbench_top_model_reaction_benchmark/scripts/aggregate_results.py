#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
LEADING_MLIPS = ROOT.parent / "leading_mlips"


MODEL_PROVENANCE = {
    "PET-OAM-XL-protocol-check": {
        "display_name": "PET-OAM-XL",
        "source": "uPET/PET production environment used by InvDesRoute",
        "checkpoint": None,
    },
    "SevenNet-Omni-i12": {
        "display_name": "SevenNet-Omni-i12",
        "source": "SevenNet official public checkpoint",
        "checkpoint": MODELS / "checkpoint_sevennet_omni_i12.ghfast.pth",
    },
    "ORB-v3-conservative-inf-MPA": {
        "display_name": "ORB-v3-conservative-inf-MPA",
        "source": "Orbital Materials official public checkpoint; Matbench Discovery MPA configuration",
        "checkpoint": MODELS / "orb-v3-conservative-inf-mpa-20250404.ckpt",
    },
    "GRACE-3L-OAM-L": {
        "display_name": "GRACE-3L-OAM-L",
        "source": "ICAMS official GRACE SavedModel; Matbench Discovery GRACE-3L-OAM-L configuration",
        "checkpoint": MODELS / "GRACE-3L-OMAT-large-ft-AM-model.tar.gz",
    },
    "EquiformerV3-DeNS-OMat24-MPtrj-sAlex": {
        "display_name": "EquiformerV3+DeNS-OAM2",
        "source": "Atomic Architects official EquiformerV3 checkpoint; OMat24 pretraining followed by MPtrj+sAlex gradient fine-tuning",
        "checkpoint": LEADING_MLIPS / "checkpoints" / "omat24-mptrj-salex_gradient.pt",
    },
    "TECE-OAM-RRA-1.0": {
        "display_name": "TECE-OAM-RRA-1.0",
        "source": "TACE official public checkpoint; OMat24+MPtrj+sAlex training configuration",
        "checkpoint": MODELS / "TECE-OAM-RRA-1.0.pt",
    },
    "EquFlashV2-45M-OAM": {
        "display_name": "EquFlashV2-45M-OAM",
        "source": "Samsung GGNN official public checkpoint; EquFlashV2 OMat24+MPtrj+sAlex configuration",
        "checkpoint": MODELS / "equflashv2-45m-oam.pt",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_one(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


def main() -> None:
    output_path = RESULTS / "current_public_model_summary.csv"
    rows_by_model = {}
    provenance_by_model = {}
    for summary_path in sorted(RESULTS.glob("*_summary.csv")):
        if summary_path == output_path:
            continue
        row = read_one(summary_path)
        if "reaction_mae_eV" not in row:
            continue
        model_id = row["model"]
        if model_id not in MODEL_PROVENANCE:
            continue
        info = MODEL_PROVENANCE[model_id]
        row = {"model": info["display_name"], **{key: value for key, value in row.items() if key != "model"}}
        rows_by_model[model_id] = row
        checkpoint = info["checkpoint"]
        provenance_by_model[model_id] = {
            "model": info["display_name"],
            "source": info["source"],
            "checkpoint": os.path.relpath(checkpoint, ROOT) if checkpoint else "external production cache",
            "checkpoint_sha256": sha256(checkpoint) if checkpoint and checkpoint.exists() else "",
            "summary_file": str(summary_path.relative_to(ROOT)),
        }

    rows = list(rows_by_model.values())
    rows.sort(key=lambda row: float(row["reaction_mae_eV"]))
    if rows:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    provenance = [provenance_by_model[model_id] for model_id in sorted(provenance_by_model)]
    (RESULTS / "current_public_model_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
