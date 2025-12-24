# src/python/run_experiment_inceptiontime_all.py
# InceptionTime multi-frequency runner (windows -> fit -> test eval -> real eval -> save)
# Uses existing results folder naming convention (no model-specific folder suffix)

import os
import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# InceptionTime (sktime)
from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier

from src.python.load_tsc_windows import load_tsc_windows
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

# Optional: keep TF from pre-allocating all GPU memory (if you are on GPU)
# Uncomment if you see TF grabbing all VRAM
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# ==========================================
# USER-DEFINED CONSTANTS
# ==========================================

LABEL_ID = 3
FREQ_TAGS = ["w", "h", "y", "q", "m", "d"]
#FREQ_TAGS = ["d"]

NUMSERIES = None              # None for full dataset; int for debugging
TEST_SIZE = 0.2
RANDOM_STATE = 42

# InceptionTime training controls
# NOTE:
#   Values below are the *official sktime defaults*.
#   Recommended operational values for large-scale rolling-window experiments
#   are provided in comments for reference only.

N_EPOCHS = 50         # sktime default 1500
# recommended: 30–50 for large rolling-window datasets (no early stopping)

BATCH_SIZE = 64        # sktime default
# recommended: 128 if memory allows (faster convergence on large datasets)

VERBOSE = True        # sktime default
# recommended: True / 1 during long runs for progress visibility

USE_RESIDUAL = True   # sktime default
USE_BOTTLENECK = True # sktime default

DEPTH = 6              # sktime default
N_FILTERS = 32         # sktime default
BOTTLENECK_SIZE = 32   # sktime default

# Scale controls (critical for huge frequencies; mirrors lessons learned from ROCKET/RotF)
MAX_TRAIN_ROWS = 2_000_000      # cap train windows (None disables)
MAX_TEST_ROWS = 400_000       # cap test windows (None disables)
MAX_REAL_ROWS = None          # real is typically smaller; keep None unless needed

# ==========================================
# PATHS (anchored at project root)
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_EXPORT_DIR = PROJECT_ROOT / "data" / "export"
RESULTS_ROOT = PROJECT_ROOT / "results" / "tsc"
MODELS_ROOT = PROJECT_ROOT / "models" / "tsc"


def print_label_stats(y, tag):
    c = Counter(y)
    total = sum(c.values())
    print(f"[{tag}] Class counts: {dict(c)} (total={total})")


def match_window_length(X, T):
    Xv = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    Xv = np.asarray(Xv, dtype=np.float64)

    cur_T = Xv.shape[1]
    if cur_T == T:
        return Xv

    if cur_T > T:
        return Xv[:, :T]

    pad = np.zeros((Xv.shape[0], T - cur_T), dtype=np.float64)
    return np.hstack([Xv, pad])


def to_sktime_nested_panel(X):
    # X: (n_instances, n_timepoints)
    Xv = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    if Xv.ndim != 2:
        raise ValueError(f"Expected 2D array for windows, got shape {Xv.shape}")
    Xv = np.asarray(Xv, dtype=np.float64)
    return pd.DataFrame({"ts": list(map(pd.Series, Xv))})


def cap_stratified(X, y, max_rows, random_state):
    # Caps by sampling indices stratified by y (best effort)
    if max_rows is None:
        return X, y
    if not isinstance(max_rows, int) or max_rows <= 0:
        return X, y
    n = len(y)
    if n <= max_rows:
        return X, y

    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)

    # Allocate per-class quota proportional to frequency (at least 1 if present)
    quotas = np.floor((counts / counts.sum()) * max_rows).astype(int)
    quotas = np.maximum(quotas, 1)

    # Fix rounding to exact max_rows
    diff = max_rows - quotas.sum()
    if diff != 0:
        order = np.argsort(-counts)
        i = 0
        while diff != 0:
            k = order[i % len(order)]
            if diff > 0:
                quotas[k] += 1
                diff -= 1
            else:
                if quotas[k] > 1:
                    quotas[k] -= 1
                    diff += 1
            i += 1

    idx_keep = []
    for cls, q in zip(classes, quotas):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) <= q:
            idx_keep.extend(cls_idx.tolist())
        else:
            idx_keep.extend(rng.choice(cls_idx, size=q, replace=False).tolist())

    idx_keep = np.array(idx_keep, dtype=int)
    rng.shuffle(idx_keep)

    X_cap = X[idx_keep] if isinstance(X, np.ndarray) else X.iloc[idx_keep]
    y_cap = y[idx_keep]
    return X_cap, y_cap


def build_inception_time(T):
    # kernel_size must be <= T, otherwise convolution is invalid
    kernel_size = int(min(40, max(8, T // 2)))
    return InceptionTimeClassifier(
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        kernel_size=kernel_size,
        n_filters=N_FILTERS,
        use_residual=USE_RESIDUAL,
        use_bottleneck=USE_BOTTLENECK,
        bottleneck_size=BOTTLENECK_SIZE,
        depth=DEPTH,
        random_state=RANDOM_STATE,
        verbose=VERBOSE,
    )


def save_model(model, save_folder: Path, filename_stem: str):
    save_folder.mkdir(parents=True, exist_ok=True)

    # Preferred: sktime save() for DL estimators
    try:
        out_zip = save_folder / f"{filename_stem}.zip"
        model.save(out_zip)
        print(f"[SAVE] Saved model via sktime.save to: {out_zip}")
        return
    except Exception as e:
        print(f"[SAVE] sktime.save failed ({e}); trying joblib...")

    # Fallback: joblib (may fail for Keras objects; keep as best-effort)
    try:
        import joblib
        out_path = save_folder / f"{filename_stem}.joblib"
        joblib.dump(model, out_path)
        print(f"[SAVE] Saved model via joblib to: {out_path}")
    except Exception as e:
        print(f"[SAVE] Skipped model saving entirely: {e}")


def run_one_frequency(freq_tag: str):
    print(f"\n=== Running InceptionTime for LABEL L{LABEL_ID}, FREQ '{freq_tag}' ===")

    windows_csv = DATA_EXPORT_DIR / f"windows_tsc_l{LABEL_ID}_{freq_tag}.csv"
    real_csv = DATA_EXPORT_DIR / f"real_eval_l{LABEL_ID}_{freq_tag}_data.csv"

    if not windows_csv.exists():
        print(f"[SKIP] Missing windows dataset: {windows_csv}")
        return
    if not real_csv.exists():
        raise FileNotFoundError(f"[REAL] Missing required REAL data file: {real_csv}")

    # -------------------------------------------------------
    # 1) Load windows dataset
    # -------------------------------------------------------
    print(f"Loading WINDOWS dataset from: {windows_csv}")
    X, y = load_tsc_windows(
        windows_csv,
        numseries=NUMSERIES,
        label_col="label",
        drop_non_numeric=False,
    )
    print("Loaded:", X.shape, y.shape)
    print_label_stats(y, "ALL")

    # -------------------------------------------------------
    # 2) Train/test split
    # -------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print("Split:", X_train.shape, X_test.shape)
    print_label_stats(y_train, "TRAIN")
    print_label_stats(y_test, "TEST")

    # Optional caps (for huge frequencies)
    X_train, y_train = cap_stratified(X_train, y_train, MAX_TRAIN_ROWS, RANDOM_STATE)
    X_test, y_test = cap_stratified(X_test, y_test, MAX_TEST_ROWS, RANDOM_STATE)
    print("After caps:", X_train.shape, X_test.shape)

    # -------------------------------------------------------
    # 3) Convert to sktime nested panel
    # -------------------------------------------------------
    t0 = time.time()
    X_train_p = to_sktime_nested_panel(X_train)
    X_test_p = to_sktime_nested_panel(X_test)
    t_conv = time.time() - t0

    T = len(X_train_p.iloc[0, 0])
    print(f"Window length T = {T}")
    print(f"[TIMER] Panel conversion: {t_conv:.2f}s")

    # -------------------------------------------------------
    # 4) Build + fit model
    # -------------------------------------------------------
    model = build_inception_time(T)

    t1 = time.time()
    print("[FIT] Training InceptionTime...")
    model.fit(X_train_p, y_train)
    t_fit = time.time() - t1
    print(f"[TIMER] InceptionTime fit: {t_fit:.2f}s (n_train={len(X_train_p)}, T={T})")

    # -------------------------------------------------------
    # 5) Evaluate on TEST
    # -------------------------------------------------------
    t2 = time.time()
    print("[PREDICT] Predicting on TEST...")
    y_pred = model.predict(X_test_p)
    t_pred = time.time() - t2
    print(f"[TIMER] InceptionTime predict (TEST): {t_pred:.2f}s (n_test={len(X_test_p)})")

    results_dir = RESULTS_ROOT / f"l{LABEL_ID}_{freq_tag}"
    print(f"[REPORT] Writing TEST results to: {results_dir}")

    evaluate_and_report(
        y_true=y_test,
        y_pred=y_pred,
        model_name="InceptionTime",
        output_dir=results_dir,
        prefix="tsc_",
    )

    # -------------------------------------------------------
    # 6) Save model
    # -------------------------------------------------------
    save_folder = MODELS_ROOT / f"l{LABEL_ID}_{freq_tag}"
    save_model(model, save_folder, filename_stem="inceptiontime")

    # -------------------------------------------------------
    # 7) REAL evaluation (strict file name)
    # -------------------------------------------------------
    print(f"\n[REAL] Using REAL evaluation DATA file: {real_csv.name}")

    X_real, y_real = load_tsc_windows(
        real_csv,
        numseries=NUMSERIES,
        label_col="true_label",
        drop_non_numeric=False,
    )

    # Optional cap for REAL (usually not needed)
    X_real, y_real = cap_stratified(X_real, y_real, MAX_REAL_ROWS, RANDOM_STATE)

    # Enforce same window length as training
    X_real_fixed = match_window_length(X_real, T)
    X_real_p = to_sktime_nested_panel(X_real_fixed)

    t3 = time.time()
    print("[PREDICT] Predicting on REAL...")
    y_real_pred = model.predict(X_real_p)
    t_real_pred = time.time() - t3
    print(f"[TIMER] InceptionTime predict (REAL): {t_real_pred:.2f}s (n_real={len(X_real_p)})")

    evaluate_and_report(
        y_true=y_real,
        y_pred=y_real_pred,
        model_name="InceptionTime_REAL",
        output_dir=results_dir,
        prefix="tsc_",
    )

    print("\n=== Frequency Completed ===")


def main():
    print(f"\n=== Running MULTI-FREQ InceptionTime for LABEL L{LABEL_ID} ===")
    print(f"Frequencies: {FREQ_TAGS}")
    print(f"NUMSERIES  : {NUMSERIES}")
    print(f"TEST_SIZE  : {TEST_SIZE}")
    print(f"SEED       : {RANDOM_STATE}")
    print(f"EPOCHS     : {N_EPOCHS}")
    print(f"BATCH_SIZE : {BATCH_SIZE}")
    print(f"MAX_TRAIN  : {MAX_TRAIN_ROWS}")
    print(f"MAX_TEST   : {MAX_TEST_ROWS}")
    print(f"MAX_REAL   : {MAX_REAL_ROWS}")

    for freq_tag in FREQ_TAGS:
        run_one_frequency(freq_tag)

    print("\n=== All Frequencies Completed ===")


if __name__ == "__main__":
    main()
