#!/usr/bin/env bash

set -e

echo "Starting full sweep..."
python experiments/sweep_runner.py

echo "Generating phase diagrams..."
python analysis/phase_diagram.py

echo "Fitting Arrhenius scaling..."
python analysis/fit_arrhenius.py

echo "All done. Results in runs/"
