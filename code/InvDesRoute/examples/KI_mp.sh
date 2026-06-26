#!/bin/sh
# Requires MP_API_KEY in the environment.
python \
  <InvDesRoute_repo>/run_screening.py \
  --target-formula KI \
  --database mp \
  --mp-max-phases 120 \
  --output-dir <InvDesRoute_repo>/results/KI_mp
