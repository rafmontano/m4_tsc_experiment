# src/python/run_experiment_dtw_only.py
# DTW-only experiment runner with timing, sanity checks, model saving, and REAL evaluation
# Uses existing results folder naming convention (no "_DTW" suffix)

import os
import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

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
FREQ_TAGS = ["w"]              # e.g. ["w","h","d","y","q","m"]
# FREQ_TAGS = ["w", "h", "d", "y", "q", "m"]

NUMSERIES = None              # None for full dataset; int for debugging
TEST_SIZE = 0.2
RANDOM_STATE = 42

# DTW controls
DTW_WINDOW = 0.10              # Sakoe-Chiba window as float (required by sktime DTW)
N_JOBS = 1                     # Recommend 1 for stability at scale (avoid joblib temp blowups)
ALGORITHM = "brute_incr"       # Avoid full distance matrix precompute

# Chunking for large prediction sets
PREDICT_CHUNK_THRESHOLD = 50_000
PREDICT_CHUNK_SIZE = 25_000
SKIP_TEST_EVAL = True


# ==========================================
# PATHS (anchored at project root)
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_EXPORT_DIR = PROJECT_ROOT / "data" / "export"
RESULTS_ROOT = PROJECT_ROOT / "results" / "tsc"
MODELS_ROOT = PROJECT_ROOT / "models" / "tsc"

# Joblib temp folder (prevents /tmp from filling up)
os.environ["JOBLIB_TEMP_FOLDER"] = str(PROJECT_ROOT / "tmp_joblib")
(PROJECT_ROOT / "tmp_joblib").mkdir(parents=True, exist_ok=True)


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
    # Accept numpy array or pandas DataFrame of shape (n_instances, n_timepoints)
    if isinstance(X, pd.DataFrame):
        Xv = X.to_numpy()
    else:
        Xv = np.asarray(X)

    if Xv.ndim != 2:
        raise ValueError(f"Expected 2D array for windows, got shape {Xv.shape}")

    # Ensure numeric (float64) once
    Xv = np.asarray(Xv, dtype=np.float64)

    # Build 1-column nested DataFrame where each cell is a pd.Series
    return pd.DataFrame({"ts": list(map(pd.Series, Xv))})


def print_label_stats(y, tag):
    c = Counter(y)
    total = sum(c.values())
    print(f"[{tag}] Class counts: {dict(c)} (total={total})")


def predict_in_chunks(model, X_panel, chunk_size=PREDICT_CHUNK_SIZE):
    preds = []
    n = len(X_panel)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        print(f"[PREDICT] rows {start:,}:{end:,}")
        preds.append(model.predict(X_panel.iloc[start:end]))
    return np.concatenate(preds)


def run_one_frequency(freq_tag: str):
    print(f"\n=== DTW-only for LABEL L{LABEL_ID}, FREQ '{freq_tag}' ===")

    # -------------------------------------------------------
    # 1) WINDOW DATASET (train/test on windows)
    # -------------------------------------------------------
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

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print("Split:", X_train.shape, X_test.shape)
    print_label_stats(y_train, "TRAIN")
    print_label_stats(y_test, "TEST")

    # -------------------------------------------------------
    # 2) Convert to sktime nested panel
    # -------------------------------------------------------
    t0 = time.time()
    X_train_p = to_sktime_nested_panel(X_train)
    X_test_p = to_sktime_nested_panel(X_test)
    t_conv = time.time() - t0

    T = len(X_train_p.iloc[0, 0])
    print(f"Window length T = {T}")
    print(f"[TIMER] Panel conversion: {t_conv:.2f}s")

    # -------------------------------------------------------
    # 3) Init DTW 1-NN model
    # -------------------------------------------------------
    dtw_window = float(DTW_WINDOW)

    dtw = KNeighborsTimeSeriesClassifier(
        n_neighbors=1,
        weights="uniform",
        distance="dtw",
        distance_params={"window": dtw_window},
        algorithm=ALGORITHM,
        n_jobs=N_JOBS,
    )

    # -------------------------------------------------------
    # 4) Fit
    # -------------------------------------------------------
    t1 = time.time()
    dtw.fit(X_train_p, y_train)
    t_fit = time.time() - t1
    print(
        f"[TIMER] DTW fit: {t_fit:.2f}s "
        f"(n_train={len(X_train_p)}, T={T}, window={dtw_window}, n_jobs={N_JOBS})"
    )

    if not SKIP_TEST_EVAL:
        # -------------------------------------------------------
        # 5) Predict on TEST (chunk if large)
        # -------------------------------------------------------
        t2 = time.time()
        if len(X_test_p) > PREDICT_CHUNK_THRESHOLD:
            y_pred = predict_in_chunks(dtw, X_test_p, chunk_size=PREDICT_CHUNK_SIZE)
        else:
            y_pred = dtw.predict(X_test_p)
        t_pred = time.time() - t2
        print(f"[TIMER] DTW predict (TEST): {t_pred:.2f}s (n_test={len(X_test_p)})")

        # -------------------------------------------------------
        # 6) Write TEST reports (no extra folder suffix)
        # -------------------------------------------------------
        results_dir = RESULTS_ROOT / f"l{LABEL_ID}_{freq_tag}"
        print(f"Writing results to: {results_dir}")

        evaluate_and_report(
            y_true=y_test,
            y_pred=y_pred,
            model_name="DTW",
            output_dir=results_dir,
            prefix="tsc_",
        )
    else:
        results_dir = RESULTS_ROOT / f"l{LABEL_ID}_{freq_tag}"
        print("[SKIP] Skipping TEST prediction/evaluation (DTW). Proceeding to REAL...")
    # -------------------------------------------------------
    # 7) Save trained DTW model (no folder suffix)
    # -------------------------------------------------------
    save_folder = MODELS_ROOT / f"l{LABEL_ID}_{freq_tag}"
    save_folder.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
        out_path = save_folder / "dtw_1nn.joblib"
        joblib.dump(dtw, out_path)
        print(f"[SAVE] Saved model to: {out_path}")
    except Exception as e:
        print(f"[SAVE] Skipped model saving: {e}")

    # -------------------------------------------------------
    # 8) REAL evaluation (strict file name; no fallback)
    # -------------------------------------------------------
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

    # Force REAL windows to match training window length T
    X_real_fixed = match_window_length(X_real, T)
    X_real_p = to_sktime_nested_panel(X_real_fixed)

    t3 = time.time()
    if len(X_real_p) > PREDICT_CHUNK_THRESHOLD:
        y_real_pred = predict_in_chunks(dtw, X_real_p, chunk_size=PREDICT_CHUNK_SIZE)
    else:
        y_real_pred = dtw.predict(X_real_p)
    t_real_pred = time.time() - t3
    print(f"[TIMER] DTW predict (REAL): {t_real_pred:.2f}s (n_real={len(X_real_p)})")

    evaluate_and_report(
        y_true=y_real,
        y_pred=y_real_pred,
        model_name="DTW_REAL",
        output_dir=results_dir,
        prefix="tsc_",
    )

    print("\n=== Frequency Completed ===")


def main():
    print(f"\n=== Running DTW-only experiment for LABEL L{LABEL_ID} ===")
    print(f"Frequencies: {FREQ_TAGS}")
    print(f"NUMSERIES  : {NUMSERIES}")
    print(f"TEST_SIZE  : {TEST_SIZE}")
    print(f"SEED       : {RANDOM_STATE}")
    print(f"DTW_WINDOW : {DTW_WINDOW}")
    print(f"N_JOBS     : {N_JOBS}")
    print(f"ALGORITHM  : {ALGORITHM}")
    print(f"CHUNK_TH   : {PREDICT_CHUNK_THRESHOLD}")
    print(f"CHUNK_SIZE : {PREDICT_CHUNK_SIZE}")

    for freq_tag in FREQ_TAGS:
        run_one_frequency(freq_tag)

    print("\n=== All Frequencies Completed ===")


if __name__ == "__main__":
    main()
