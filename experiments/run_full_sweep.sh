#!/usr/bin/env bash

set -e

echo "Starting full sweep..."
python experiments/sweep_runner.py
python lambda_sweep.py

echo "Generating phase diagrams..."
python analysis/phase_diagram.py

echo "Fitting Arrhenius scaling..."
python analysis/fit_arrhenius.py

echo "Creating precision reallocation figure..."
python analysis/fir_precision.py

echo "All done. Results in runs/"
