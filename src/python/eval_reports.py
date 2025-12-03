# File: src/python/eval_reports.py
# Purpose:
#   Generic evaluation helper for classification models.
#   Computes accuracy, kappa, per-class precision/recall/F1, macro F1,
#   and saves Caret-like outputs (confusion matrix, metrics, JSON summary,
#   and sklearn text report) under results/.

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    cohen_kappa_score,
)


def evaluate_and_report(
    y_true,
    y_pred,
    model_name: str,
    output_dir: str | Path = "results/python",
    prefix: str = "",
    verbose: bool = True,
):
    """
    Compute metrics and save reports in a Caret-like style.

    Parameters
    ----------
    y_true : array-like
        True class labels (e.g. 0, 1, 2).
    y_pred : array-like
        Predicted class labels.
    model_name : str
        Short model identifier (e.g. "ROCKET", "InceptionTime").
    output_dir : str or Path, default "results/python"
        Directory where all output files will be written.
    prefix : str, optional
        Optional prefix for all filenames (e.g. "l3_q_").
    verbose : bool, default True
        If True, print metrics and file locations to stdout.

    Files created in output_dir:
      - <prefix><model_name>_confusion.csv
      - <prefix><model_name>_metrics.csv
      - <prefix><model_name>_metrics.json
      - <prefix><model_name>_classification_report.txt
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    precisions, recalls, f1s, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    classes = np.unique(y_true)
    macro_f1 = float(f1s.mean())

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = "Predicted"
    cm_df.columns.name = "True"

    # Per-class metrics (shape similar to R metrics tables)
    metrics_df = pd.DataFrame(
        {
            "class": classes,
            "precision": precisions,
            "recall": recalls,
            "f1": f1s,
        }
    )
    metrics_df["overall_accuracy"] = float(acc)
    metrics_df["kappa"] = float(kappa)
    metrics_df["macro_f1"] = float(macro_f1)
    metrics_df["method"] = model_name

    # ---------- build file paths ----------
    stem = f"{prefix}{model_name}"

    cm_path = output_dir / f"{stem}_confusion.csv"
    metrics_csv_path = output_dir / f"{stem}_metrics.csv"
    metrics_json_path = output_dir / f"{stem}_metrics.json"
    report_path = output_dir / f"{stem}_classification_report.txt"
    # -------------------------------------

    # 1) confusion matrix CSV
    cm_df.to_csv(cm_path)

    # 2) per-class metrics CSV
    metrics_df.to_csv(metrics_csv_path, index=False)

    # 3) summary JSON
    summary = {
        "model": model_name,
        "overall_accuracy": float(acc),
        "kappa": float(kappa),
        "macro_f1": float(macro_f1),
        "classes": classes.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    with open(metrics_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 4) sklearn text report
    report_str = classification_report(y_true, y_pred, digits=4)
    with open(report_path, "w") as f:
        f.write(report_str + "\n")

    if verbose:
        print("\n====================== Evaluation (Python) ======================")
        print(f"Model       : {model_name}")
        print(f"Accuracy    : {acc:.4f}")
        print(f"Kappa       : {kappa:.4f}")
        print(f"Macro F1    : {macro_f1:.4f}")
        print("\nConfusion Matrix:")
        print(cm_df)
        print("\nPer-class metrics:")
        print(metrics_df[["class", "precision", "recall", "f1"]])
        print("\nsklearn classification_report:")
        print(report_str)
        print("\nSaved:")
        print(f"  Confusion CSV : {cm_path}")
        print(f"  Metrics CSV   : {metrics_csv_path}")
        print(f"  Metrics JSON  : {metrics_json_path}")
        print(f"  Report TXT    : {report_path}")

    return {
        "accuracy": float(acc),
        "kappa": float(kappa),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
        "metrics_df": metrics_df,
    }