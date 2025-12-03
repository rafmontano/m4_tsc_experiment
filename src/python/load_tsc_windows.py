# src/load_tsc_windows.py

import numpy as np
import pandas as pd
from pathlib import Path


def load_tsc_windows(
    csv_path,
    numseries=None,
    label_col="label",
    drop_non_numeric=False,
):
    """
    Load TSC windows (or REAL eval) from a CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV.
    numseries : int or None
        If not None, use only the first `numseries` rows (for debugging).
    label_col : str
        Name of the label column. For training windows this is usually "label".
        For REAL eval from R/11b_eval_real_tsc.R this is "true_label".
    drop_non_numeric : bool
        If True: keep only numeric columns and then drop `label_col` if present.
                 Use this for REAL eval datasets where you have metadata
                 columns such as 'st'.
        If False: drop only the label column and keep the rest as features.
                  Use this for window/train/test datasets.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features), dtype float32
    y : np.ndarray, shape (n_samples,), dtype int64
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise KeyError(
            f"Label column '{label_col}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # 1) Extract labels and robustly coerce to int64
    y_raw = df[label_col].to_numpy()
    y_str = pd.Series(y_raw, dtype="object").astype(str).str.strip()

    try:
        y = y_str.astype("int64").to_numpy()
    except ValueError:
        y_numeric_part = y_str.str.extract(r"(-?\d+)")[0]
        bad_mask = y_numeric_part.isna()
        if bad_mask.any():
            bad_examples = y_str[bad_mask].unique()[:10]
            raise ValueError(
                "Failed to parse some labels to integers, even after cleaning.\n"
                f"Example problematic labels: {bad_examples}"
            )
        y = y_numeric_part.astype("int64").to_numpy()

    # 2) Build feature matrix DataFrame
    if drop_non_numeric:
        # REAL eval: keep only numeric columns, then drop label_col if present
        feature_df = df.select_dtypes(include=["number"]).copy()
        if label_col in feature_df.columns:
            feature_df = feature_df.drop(columns=[label_col])
    else:
        # Window dataset: start with all columns except label_col
        feature_df = df.drop(columns=[label_col])

    # 3) Coerce all features to numeric, handle garbage values
    coerced_df = feature_df.apply(pd.to_numeric, errors="coerce")

    n_na = coerced_df.isna().sum().sum()
    if n_na > 0:
        # For now, replace with 0.0 and continue
        print(
            f"[load_tsc_windows] Warning: {n_na} non-numeric feature values "
            f"were coerced to NaN and filled with 0.0."
        )
        coerced_df = coerced_df.fillna(0.0)

    X = coerced_df.to_numpy(dtype=np.float32)

    # 4) Optional row downsampling for debugging
    if numseries is not None:
        X = X[:numseries]
        y = y[:numseries]

    return X, y