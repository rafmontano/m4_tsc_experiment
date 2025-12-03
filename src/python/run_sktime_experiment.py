# src/python/run_sktime_experiment.py

import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path

from src.python.eval_reports import evaluate_and_report


def run_sktime_experiment(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    output_dir: Path,
    prefix: str = "tsc_",
):
    """
    Trains each model, evaluates on the test set, and writes reports.

    Returns
    -------
    results : dict
        {model_name: accuracy}
    fitted_models : dict
        {model_name: fitted_estimator}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure arrays are clean int64 1D vectors
    y_train = np.asarray(y_train).astype(np.int64).ravel()
    y_test = np.asarray(y_test).astype(np.int64).ravel()

    results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        fitted_models[name] = model

        print(f"{name} Accuracy: {acc:.4f}")

        # Write confusion, metrics, report
        evaluate_and_report(
            y_true=y_test,
            y_pred=y_pred,
            model_name=name,
            output_dir=output_dir,
            prefix=prefix,
        )

    return results, fitted_models