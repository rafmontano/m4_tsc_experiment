# src/python/run_experiment_rotf_all.py
# End-to-end RotF runner using FFORMA-style exported predictors (tabular)

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sktime.classification.sklearn import RotationForest

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

LABEL_ID = 3
#FREQ_TAGS = ["w", "h", "d", "y", "q", "m"]   # adjust as needed
FREQ_TAGS = [  "m", "d", "q"]   # adjust as needed
NUMSERIES = None                         #100     # None for full dataset; int for debugging
RANDOM_STATE = 42
ROTF_MAX_ROWS = 4_000_000   # adjust (e.g., 1_000_000, 2_000_000, 4_000_000)

# RotF parameters (start conservative, then tune)
N_ESTIMATORS = 50
N_JOBS = 16 #-1

# ==========================================
# PATHS (anchored at project root)
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_EXPORT_DIR = PROJECT_ROOT / "data" / "export"
RESULTS_ROOT = PROJECT_ROOT / "results" / "tsc"
MODELS_ROOT = PROJECT_ROOT / "models" / "tsc"

LABEL_CANDIDATES = ["label", "true_label", "y", "target", "class"]
DROP_ID_CANDIDATES = ["id", "series_id", "window_id", "idx", "index", ".id"]

###
os.environ["JOBLIB_TEMP_FOLDER"] = str(PROJECT_ROOT / "tmp_joblib")
(PROJECT_ROOT / "tmp_joblib").mkdir(parents=True, exist_ok=True)




def _detect_label_col(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"No label column found. Tried {LABEL_CANDIDATES}. "
        f"Columns include: {list(df.columns)[:30]}"
    )


def _load_tabular_xy(csv_path: Path, impute_medians=None, return_medians=False):
    df = pd.read_csv(csv_path)

    # Cap rows (priority: NUMSERIES for debugging, else ROTF_MAX_ROWS safety cap)
    if isinstance(NUMSERIES, int) and NUMSERIES > 0 and len(df) > NUMSERIES:
        df = df.sample(n=NUMSERIES, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[CAP] {csv_path.name}: capped to {len(df)} rows")
    elif isinstance(ROTF_MAX_ROWS, int) and ROTF_MAX_ROWS > 0 and len(df) > ROTF_MAX_ROWS:
        df = df.sample(n=ROTF_MAX_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[CAP] {csv_path.name}: capped to {len(df)} rows")

    label_col = _detect_label_col(df)
    y = df[label_col].to_numpy()

    drop_cols = set([label_col])
    for c in DROP_ID_CANDIDATES:
        if c in df.columns:
            drop_cols.add(c)

    X = df.drop(columns=list(drop_cols), errors="ignore")

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if impute_medians is None:
        impute_medians = X.median(numeric_only=True)

    X = X.fillna(impute_medians)

    all_nan_cols = X.columns[X.isna().all()].tolist()
    if len(all_nan_cols) > 0:
        X = X.drop(columns=all_nan_cols)

    X_np = X.to_numpy(dtype=np.float64)

    if return_medians:
        return X_np, y, impute_medians
    return X_np, y




def run_one_frequency(freq_tag: str):
    print(f"\n=== Running RotF (FFORMA predictors) for LABEL L{LABEL_ID}, FREQ '{freq_tag}' ===")

    train_csv = DATA_EXPORT_DIR / f"train_l{LABEL_ID}_{freq_tag}.csv"
    test_csv  = DATA_EXPORT_DIR / f"test_l{LABEL_ID}_{freq_tag}.csv"

    if not train_csv.exists():
        print(f"[SKIP] Train file not found: {train_csv}")
        return
    if not test_csv.exists():
        print(f"[SKIP] Test file not found: {test_csv}")
        return

    # -------------------------------------------------------
    # 1) Load TRAIN, compute medians for safe imputation
    # -------------------------------------------------------
    print(f"Loading TRAIN predictors from: {train_csv}")
    X_train, y_train, train_medians = _load_tabular_xy(train_csv, return_medians=True)

    # -------------------------------------------------------
    # 2) Load TEST using TRAIN medians (no leakage)
    # -------------------------------------------------------
    print(f"Loading TEST predictors from : {test_csv}")
    X_test, y_test = _load_tabular_xy(test_csv, impute_medians=train_medians)

    print("Shapes:", X_train.shape, y_train.shape, "|", X_test.shape, y_test.shape)

    # -------------------------------------------------------
    # 3) Train RotF
    # -------------------------------------------------------
    model = RotationForest(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    print("\n[FIT] Training RotF...")
    model.fit(X_train, y_train)

    # -------------------------------------------------------
    # 4) Evaluate on TEST
    # -------------------------------------------------------
    print("[PREDICT] Predicting RotF on TEST...")
    y_pred = model.predict(X_test)

    results_dir = RESULTS_ROOT / f"l{LABEL_ID}_{freq_tag}"
    print(f"[REPORT] Writing results to: {results_dir}")

    evaluate_and_report(
        y_true=y_test,
        y_pred=y_pred,
        model_name="RotF",
        output_dir=results_dir,
        prefix="tsc_",
    )

    # -------------------------------------------------------
    # 5) Save model
    # -------------------------------------------------------
    save_folder = MODELS_ROOT / f"l{LABEL_ID}_{freq_tag}"
    save_folder.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
        out_path = save_folder / "rotf.joblib"
        joblib.dump(model, out_path)
        print(f"[SAVE] Saved model to: {out_path}")
    except Exception as e:
        print(f"[SAVE] Skipped model saving: {e}")

    # -------------------------------------------------------
    # 6) REAL evaluation (NO fallback; strict file name)
    # -------------------------------------------------------
    real_data_path = DATA_EXPORT_DIR / f"real_eval_l{LABEL_ID}_{freq_tag}_data.csv"
    if not real_data_path.exists():
        raise FileNotFoundError(f"[REAL] Missing required REAL data file: {real_data_path}")

    print(f"\n[REAL] Using REAL evaluation DATA file: {real_data_path.name}")

    X_real, y_real = _load_tabular_xy(real_data_path, impute_medians=train_medians)

    print("[PREDICT] Predicting RotF on REAL...")
    y_real_pred = model.predict(X_real)

    evaluate_and_report(
        y_true=y_real,
        y_pred=y_real_pred,
        model_name="RotF_REAL",
        output_dir=results_dir,
        prefix="tsc_",
    )

    print("\n=== Frequency Completed ===")



def main():
    print(f"\n=== Running MULTI-FREQ RotF (FFORMA predictors) for LABEL L{LABEL_ID} ===")
    print(f"Frequencies: {FREQ_TAGS}")
    print(f"NUMSERIES  : {NUMSERIES}")
    print(f"SEED       : {RANDOM_STATE}")
    print(f"RotF       : n_estimators={N_ESTIMATORS}, n_jobs={N_JOBS}")

    for freq_tag in FREQ_TAGS:
        run_one_frequency(freq_tag)

    print("\n=== All Frequencies Completed ===")


if __name__ == "__main__":
    main()
