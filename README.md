# Grokking as Metastable Complexity Dynamics

Author: Matthew Lakatos
Contact: m.atthew.lakatos1@gmail.com

## Overview
This repository contains code and analysis for the paper *Grokking as Metastable Complexity Dynamics*. The project formalises grokking as a metastable escape in a complexity-regularised free-energy landscape and provides a minimal, reproducible experimental protocol to validate the theory.

## Quick start (smoke test)
Run the canonical smoke experiment and produce the main trajectory figure:

```bash
# create env, install requirements
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# run the smoke experiment (config in configs/modular_addition.yaml)
./reproducibility/reproduce.sh
