# InvDesRoute code

This directory contains a cleaned snapshot of the InvDesRoute workflow code.

## Main modules

- `InvDesRoute/run_screening.py`: formula/CIF-to-route workflow entry point.
- `InvDesRoute/benchmark_mp_pet.py`: 120-target MP/PET benchmark driver.
- `InvDesRoute/aggregate_recommendations.py`: target-level recommendation aggregation.
- `InvDesRoute/build_diverse_recommendations.py`: diversity-aware route display.
- `InvDesRoute/apply_condition_corrections.py`: first-order gas-reservoir sensitivity correction.
- `InvDesRoute/literature_benchmark/`: literature-route extraction, route-family comparison and ranking-ablation scripts.
- `InvDesRoute/src/`: phase retrieval, PET energy handling, reaction enumeration, thermodynamic analysis and synthesizability scoring modules.

## Environment

A minimal conda environment file is provided as `environment.yml`. Additional model-specific environments may be needed for PET-OAM-XL/uPET and optional baseline models.

```bash
conda env create -f environment.yml
conda activate invdesroute
```

## Required external resources

- Materials Project access requires an API key in `MP_API_KEY` or `PMG_MAPI_KEY`.
- PET-OAM-XL checkpoint path should be provided through local configuration or command-line arguments. Model checkpoints are not redistributed in this release.
- Raw text-mined synthesis data should be downloaded from the public Ceder/Kononova repository if rerunning literature extraction from scratch.

## Example commands

Run a formula input with local/available phase data:

```bash
python InvDesRoute/run_screening.py --formula KI --output-dir results/KI_example
```

Run a Materials Project-backed query:

```bash
export MP_API_KEY=your_materials_project_key
python InvDesRoute/run_screening.py \
  --formula BaTiO3 \
  --use-mp \
  --output-dir results/BaTiO3_mp
```

Run with PET energy evaluation when a uPET/PET environment is available:

```bash
export MP_API_KEY=your_materials_project_key
export UPET_PYTHON=/path/to/upet_env/bin/python
export PET_OAM_XL_CHECKPOINT=/path/to/pet-oam-xl-v1.0.0.ckpt
python InvDesRoute/run_screening.py \
  --formula BaTiO3 \
  --use-mp \
  --compute-pet \
  --pet-python "$UPET_PYTHON" \
  --output-dir results/BaTiO3_pet
```

## Reproducing manuscript tables

The uploaded `data/` directory contains frozen processed tables used in the manuscript. Recomputing all tables from scratch requires Materials Project access, the PET-OAM-XL checkpoint and sufficient compute time. The scripts in this release provide the workflow logic; users should adapt local paths and model environments in their own configuration.
