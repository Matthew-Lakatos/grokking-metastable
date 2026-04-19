#!/usr/bin/env bash
# run_full_sweep.sh
# Orchestrates the complete set of experiments and analyses for the paper.
#
# Run from the repository root:
#   chmod +x run_full_sweep.sh
#   ./run_full_sweep.sh

set -e

echo "=== 1/4  Arrhenius (learning-rate) sweep ==="
python experiments/sweep_runner.py

echo "=== 2/4  Lambda (weight-decay) sweep ==="
python experiments/lambda_sweep.py

echo "=== 3/4  Dataset-size sweep ==="
python experiments/dataset_sweep.py

echo "=== 4/4  Causal test ==="
python experiments/causal_test.py

echo ""
echo "=== Generating authoritative paper figures ==="
python final_output/analyser.py

echo ""
echo "=== Optional: additional analysis scripts ==="
echo "  python analysis/fit_arrhenius.py"
echo "  python analysis/fit_precision.py --log <path/to/log_seed0.csv>"
echo "  python analysis/phase_diagram.py"

echo ""
echo "All done. Paper figures are in final_output/."
