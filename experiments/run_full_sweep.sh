#!/usr/bin/env bash
# Full sweep and analysis

echo "Starting sweep..."
python experiments/sweep_runner.py

echo "Generating phase diagrams..."
python analysis/phase_diagram.py

echo "Fitting Arrhenius scaling..."
python analysis/fit_arrhenius.py

echo "All done. Results in runs/"
