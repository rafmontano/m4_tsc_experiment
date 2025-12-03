# M4 Time Series Classification Experiment

Full Experiment Pipeline (R + Python) for Directional Forecasting Using Label 3

This repository contains the unified, end-to-end experiment pipeline used to evaluate forecasting and time-series-classification models across all M4 frequencies (yearly, quarterly, monthly, weekly, daily). The system supports R and Python, uses label 3 by default, and produces the final outputs for the research paper: • consolidated_results.xlsx • cd_diagram_models.pdf

Both of which are stored under:

results/consolidated/

This README provides installation instructions, project structure, and execution workflow.

⸻

# 1. Project Goals

This project unifies two previously separate Proof-of-Concepts: • R PoC: XGBoost directional classifier • Python PoC: ROCKET, InceptionTime, and baseline models (SMYL, FFORMA equivalents)

The objective is to run a full M4 TSC experiment and produce: 1. Model performance tables 2. A critical difference (CD) diagram 3. A reproducible paper-ready dataset

⸻

# 2. Project Folder Structure

The repository follows a clean, reproducible structure:

m4_tsc_experiment/ ├─ config/ ├─ data/ │ ├─ raw/m4/ │ ├─ processed/features/ │ ├─ processed/labels/ │ └─ metadata/ ├─ models/ ├─ src/ │ ├─ r/ │ └─ python/ ├─ scripts/ ├─ results/ │ ├─ intermediate/ │ ├─ consolidated/ │ └─ logs/ ├─ figures/ ├─ docs/ └─ 00_install.R

Key folders:

Folder Purpose config/ Model lists, frequencies, labels, experiment settings data/raw/ Raw M4 dataset (not tracked by git) data/processed/labels/ Labelled directional data models/ Saved model artefacts per label/frequency/model results/intermediate/ Predictions and metrics (large, ignored by git) results/consolidated/ Final Excel + CD diagram src/r/ R code (XGBoost, data prep, aggregation) src/python/ Python code (ROCKET, InceptionTime, baselines) scripts/ Shell/PowerShell runners docs/ Paper (.tex), figures, notes

⸻

# 3. Installation

Step 1 — Run the bootstrap script

This installs required R packages and creates all necessary folders.

source("00_install.R")

Step 2 — Create your Python environment

Example (Linux/macOS/Windows):

python -m venv .venv source .venv/bin/activate \# Windows: .venv\Scripts\activate

pip install --upgrade pip pip install -r requirements.txt \# (if present)

If requirements.txt is not created yet, typical packages include: • numpy • pandas • scikit-learn • sktime • aeon • numba • torch • matplotlib

⸻

# 4. Data Preparation

4.1 Load the M4 dataset

After installing the package via 00_install.R, run:

Rscript src/r/01_prepare_data.R

This will: • Load M4 data from data/raw/m4/ • Generate label-3 directional datasets • Save outputs to:

data/processed/labels/label_3/

⸻

# 5. Running the Experiment (Model Training)

The experiment is run independently per frequency due to computational constraints.

R Example (XGBoost)

Rscript src/r/10_train_xgboost.R --freq quarterly --label 3

Python Example (ROCKET)

python src/python/10_run_rocket.py --freq quarterly --label 3

Each run saves: • Trained model → models/label_3/<frequency>/<model>/ • Predictions + metrics → results/intermediate/label_3/<frequency>/<model>/

⸻

# 6. Running Label 3 for All Frequencies

Use the driver script:

Linux/macOS

bash scripts/run_all_label3.sh

Windows PowerShell

.\scripts\run\_all_label3.ps1

This will: 1. Run all models for all frequencies 2. Save all intermediate results 3. Produce the final tables and CD diagram

⸻

# 7. Producing the Final Experiment Outputs

After all frequencies + models have run:

Rscript scripts/make_cd_and_table.R

This script builds: • results/consolidated/consolidated_results.xlsx • results/consolidated/cd_diagram_models.pdf

These are the main artifacts for the research paper.

⸻

# 8. Paper & Documentation

All academic material lives in:

docs/paper/

Including: • main.tex • figures • methodology notes • experiment logs

This folder is structured to be compatible with Overleaf.

⸻

# 9. Reproducibility Guarantees

The pipeline is designed so that: • Raw data is cleanly separated • Model artefacts are preserved • Intermediate results are never committed • All scripts are deterministic when seeded

This ensures full reproducibility for reviewers and future experiments.

⸻

# 10. Credits

This research is part of the PhD project at the University of Sydney, focusing on: • Time series classification • Directional forecasting • Meta-learning and M4/M6 financial data
