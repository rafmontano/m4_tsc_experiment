# src/python/run_experiment.py

import os
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.python.load_tsc_windows import load_tsc_windows
from src.python.get_sktime_models import get_sktime_models
from src.python.run_sktime_experiment import run_sktime_experiment
from src.python.model_io import save_models
from src.python.eval_reports import evaluate_and_report

# --------------------------------------------------
# Silence TensorFlow and deprecation / numba chatter
# --------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from numba.core.errors import NumbaTypeSafetyWarning
    warnings.filterwarnings("ignore", category=NumbaTypeSafetyWarning)
except Exception:
    pass

# ==========================================
# USER-DEFINED CONSTANTS
# ==========================================

LABEL_ID = 3          # 1, 2, 3, 4 → L1...L4
FREQ_TAG = "y"        # e.g. "y", "q", "m", etc.
NUMSERIES = 1000 #None      # None for full dataset

# ==========================================
# PATHS (anchored at project root)
# ==========================================

# This file: .../m4_tsc_experiment/src/python/run_experiment.py
# project_root: .../m4_tsc_experiment
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_EXPORT_DIR = PROJECT_ROOT / "data" / "export"
RESULTS_ROOT = PROJECT_ROOT / "results" / "tsc"
MODELS_ROOT = PROJECT_ROOT / "models" / "tsc"


def main():
    print(f"\n=== Running TSC experiment for LABEL L{LABEL_ID}, FREQ '{FREQ_TAG}' ===")

    # -------------------------------------------------------
    # 1) WINDOW DATASET (for training/testing on windows)
    # -------------------------------------------------------
    csv_file = DATA_EXPORT_DIR / f"windows_tsc_l{LABEL_ID}_{FREQ_TAG}.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_file}")

    print(f"Loading dataset from: {csv_file}")
    X, y = load_tsc_windows(
        csv_file,
        numseries=NUMSERIES,
        label_col="label",          # window files use 'label'
        drop_non_numeric=False,     # all feature cols are predictors
    )
    print("Loaded:", X.shape, y.shape)

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -------------------------------------------------------
    # 2) Init TSC models
    # -------------------------------------------------------
    models = get_sktime_models()

    # -------------------------------------------------------
    # 3) Train + evaluate on windows + write reports
    # -------------------------------------------------------
    results_dir = RESULTS_ROOT / f"l{LABEL_ID}_{FREQ_TAG}"
    print(f"Results will be written to: {results_dir}")

    results, fitted_models = run_sktime_experiment(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=results_dir,
        prefix="tsc_",   # tsc_1NN_ED_..., tsc_ROCKET_..., etc.
    )

    print("\n=== Accuracy Matrix ===")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")

    # -------------------------------------------------------
    # 4) Save trained models
    # -------------------------------------------------------
    save_folder = MODELS_ROOT / f"l{LABEL_ID}_{FREQ_TAG}"
    print(f"\nSaving models to: {save_folder}")
    save_models(fitted_models, folder=save_folder)

    # -------------------------------------------------------
    # 5) REAL last-window validation (Python version of step 11b)
    # -------------------------------------------------------
    real_csv = DATA_EXPORT_DIR / f"real_eval_tsc_l{LABEL_ID}_{FREQ_TAG}.csv"

    if real_csv.exists():
        print(f"\n[REAL] Found REAL TSC eval file: {real_csv.name}")
        print("[REAL] Loading REAL last-window dataset...")

        X_real, y_real = load_tsc_windows(
            real_csv,
            numseries=None,
            label_col="true_label",   # matches 11b_eval_real_tsc.R
            drop_non_numeric=True,    # drop id, keep only numeric window columns
        )

        print("[REAL] Shapes:", X_real.shape, y_real.shape)

        for name, model in fitted_models.items():
            print(f"\n[REAL] Evaluating model {name} on REAL last-window data...")

            y_real_pred = model.predict(X_real)

            evaluate_and_report(
                y_true=y_real,
                y_pred=y_real_pred,
                model_name=f"{name}_REAL",
                output_dir=results_dir,
                prefix="tsc_",
            )
    else:
        print(f"\n[REAL] No REAL eval file found at {real_csv}")

    print("\n=== Experiment Completed ===")


if __name__ == "__main__":
    main()