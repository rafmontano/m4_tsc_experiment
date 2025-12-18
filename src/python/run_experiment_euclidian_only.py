# src/python/run_experiment_euclidean2.py
# Euclidean-only (1-NN) experiment runner with timing, sanity checks, model saving, and REAL evaluation
# Uses existing results folder naming convention (no "_EUCLIDEAN" suffix)

import os
import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

# ==========================================
# USER-DEFINED CONSTANTS
# ==========================================

LABEL_ID = 3
#FREQ_TAGS = ["h", "d", "y", "q", "m"]  # adjust as needed
#FREQ_TAGS = ["w", "h", "y", "q", "m", "d"]
FREQ_TAGS = [ "d", "m"]

NUMSERIES = None
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cap TRAIN size to avoid OOM (None = no cap)
MAX_TRAIN_SERIES = 2_000_000   # start conservative; tune upward
MAX_TEST_SERIES  = None        # optional cap if needed
MAX_REAL_SERIES  = None        # optional cap if needed


# sklearn KNN controls
# Note: sklearn KNeighborsClassifier supports n_jobs in recent versions; if your version ignores it, it will still work.
N_JOBS = 16

# Prediction verbosity / chunking
SKIP_TEST_EVAL = True
PREDICT_CHUNK_SIZE = 10000  # per your request

# Use float32 to cut memory in half (usually fine for Euclidean distance)
DTYPE = np.float32

# ==========================================
# PATHS (anchored at project root)
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_EXPORT_DIR = PROJECT_ROOT / "data" / "export"
RESULTS_ROOT = PROJECT_ROOT / "results" / "tsc"
MODELS_ROOT = PROJECT_ROOT / "models" / "tsc"


def cap_rows(X, y, max_rows, seed, tag):
    if max_rows is None:
        return X, y

    n = len(y)
    if n <= max_rows:
        return X, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_rows, replace=False)

    # Works for numpy arrays and pandas dataframes
    if isinstance(X, pd.DataFrame):
        X_cap = X.iloc[idx].reset_index(drop=True)
    else:
        X_cap = X[idx]

    y_cap = np.asarray(y)[idx]
    print(f"[CAP] {tag}: capped from {n:,} to {len(y_cap):,} rows")
    return X_cap, y_cap



def print_label_stats(y, tag):
    c = Counter(y)
    total = sum(c.values())
    print(f"[{tag}] Class counts: {dict(c)} (total={total})")


def match_window_length_np(X, T):
    Xv = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    Xv = np.asarray(Xv, dtype=DTYPE)

    cur_T = Xv.shape[1]
    if cur_T == T:
        return Xv
    if cur_T > T:
        return Xv[:, :T]

    pad = np.zeros((Xv.shape[0], T - cur_T), dtype=DTYPE)
    return np.hstack([Xv, pad])


def predict_in_chunks_verbose(model, X_np, chunk_size=PREDICT_CHUNK_SIZE, tag="PREDICT"):
    n = X_np.shape[0]
    out = np.empty((n,), dtype=np.int64)

    t0 = time.time()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        t_chunk = time.time()
        out[start:end] = model.predict(X_np[start:end])
        dt = time.time() - t_chunk

        done = end
        rate = (end - start) / max(dt, 1e-9)
        elapsed = time.time() - t0
        print(f"[{tag}] rows {start:,}:{end:,} ({done:,}/{n:,})  chunk={dt:.2f}s  rate={rate:,.0f} rows/s  elapsed={elapsed:.1f}s")

    return out


def run_one_frequency(freq_tag: str):
    print(f"\n=== EUCLIDEAN-only for LABEL L{LABEL_ID}, FREQ '{freq_tag}' ===")

    csv_file = DATA_EXPORT_DIR / f"windows_tsc_l{LABEL_ID}_{freq_tag}.csv"
    if not csv_file.exists():
        print(f"[SKIP] Dataset not found: {csv_file}")
        return

    print(f"Loading dataset from: {csv_file}")
    X, y = load_tsc_windows(
        csv_file,
        numseries=NUMSERIES,
        label_col="label",
        drop_non_numeric=False,
    )
    print("Loaded:", X.shape, y.shape)
    print_label_stats(y, "ALL")

    # Convert once to dense numeric matrix
    X_np = np.asarray(X, dtype=DTYPE)
    T = X_np.shape[1]
    print(f"Window length T = {T}")

    # Split first (no caps yet)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Cap TRAIN only (prevents train OOM, avoids touching test distribution)
    X_train, y_train = cap_rows(
        X_train, y_train, MAX_TRAIN_SERIES, RANDOM_STATE, tag="TRAIN"
    )

    # Optional caps (only if needed)
    X_test, y_test = cap_rows(
        X_test, y_test, MAX_TEST_SERIES, RANDOM_STATE, tag="TEST"
    )

    print("Split:", X_train.shape, X_test.shape)
    print_label_stats(y_train, "TRAIN")
    print_label_stats(y_test, "TEST")

    # 1-NN Euclidean
    eucl = KNeighborsClassifier(
        n_neighbors=1,
        weights="uniform",
        metric="euclidean",
        n_jobs=N_JOBS,
    )

    t1 = time.time()
    eucl.fit(X_train, y_train)
    t_fit = time.time() - t1
    print(f"[TIMER] Euclidean fit: {t_fit:.2f}s (n_train={len(X_train):,}, T={T}, n_jobs={N_JOBS})")

    results_dir = RESULTS_ROOT / f"l{LABEL_ID}_{freq_tag}"

    if not SKIP_TEST_EVAL:
        t2 = time.time()
        y_pred = predict_in_chunks_verbose(eucl, X_test, chunk_size=PREDICT_CHUNK_SIZE, tag="PREDICT_TEST")
        t_pred = time.time() - t2
        print(f"[TIMER] Euclidean predict (TEST): {t_pred:.2f}s (n_test={len(X_test):,})")

        print(f"Writing results to: {results_dir}")
        evaluate_and_report(
            y_true=y_test,
            y_pred=y_pred,
            model_name="EUCLIDEAN",
            output_dir=results_dir,
            prefix="tsc_",
        )
    else:
        print("[SKIP] Skipping TEST prediction/evaluation (EUCLIDEAN). Proceeding to REAL...")

    # Save model
    save_folder = MODELS_ROOT / f"l{LABEL_ID}_{freq_tag}"
    save_folder.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        out_path = save_folder / "euclidean_1nn.joblib"
        joblib.dump(eucl, out_path)
        print(f"[SAVE] Saved model to: {out_path}")
    except Exception as e:
        print(f"[SAVE] Skipped model saving: {e}")

    # REAL evaluation
    real_data_path = DATA_EXPORT_DIR / f"real_eval_l{LABEL_ID}_{freq_tag}_data.csv"
    if not real_data_path.exists():
        raise FileNotFoundError(f"[REAL] Missing required REAL data file: {real_data_path}")

    print(f"\n[REAL] Using REAL evaluation DATA file: {real_data_path.name}")

    X_real, y_real = load_tsc_windows(
        real_data_path,
        numseries=NUMSERIES,
        label_col="true_label",
        drop_non_numeric=False,
    )

    X_real_np = match_window_length_np(X_real, T)

    t3 = time.time()
    y_real_pred = predict_in_chunks_verbose(eucl, X_real_np, chunk_size=PREDICT_CHUNK_SIZE, tag="PREDICT_REAL")
    t_real_pred = time.time() - t3
    print(f"[TIMER] Euclidean predict (REAL): {t_real_pred:.2f}s (n_real={len(X_real_np):,})")

    evaluate_and_report(
        y_true=y_real,
        y_pred=y_real_pred,
        model_name="EUCLIDEAN_REAL",
        output_dir=results_dir,
        prefix="tsc_",
    )

    print("\n=== Frequency Completed ===")


def main():
    print(f"\n=== Running EUCLIDEAN-only experiment for LABEL L{LABEL_ID} ===")
    print(f"Frequencies: {FREQ_TAGS}")
    print(f"NUMSERIES  : {NUMSERIES}")
    print(f"TEST_SIZE  : {TEST_SIZE}")
    print(f"SEED       : {RANDOM_STATE}")
    print(f"N_JOBS     : {N_JOBS}")
    print(f"CHUNK_SIZE : {PREDICT_CHUNK_SIZE}")
    print(f"SKIP_TEST  : {SKIP_TEST_EVAL}")
    print(f"DTYPE      : {DTYPE}")

    for freq_tag in FREQ_TAGS:
        run_one_frequency(freq_tag)

    print("\n=== All Frequencies Completed ===")


if __name__ == "__main__":
    main()
