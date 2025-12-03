# src/model_io.py

from pathlib import Path
import joblib


def save_models(models: dict, folder: str | Path):
    """
    Save a dictionary of fitted sktime models to disk using joblib.

    Parameters
    ----------
    models : dict
        {model_name: fitted_model}

    folder : str or Path
        Directory where <model_name>.joblib should be written.
        RECOMMENDED (new project structure):
            models/sktime/<label>/<frequency>/
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        path = folder / f"{name}.joblib"
        joblib.dump(model, path)
        print(f"[model_io] Saved {name} → {path}")


def load_models(folder: str | Path, names: list[str]) -> dict:
    """
    Load previously saved sktime TSC models.

    Parameters
    ----------
    folder : str or Path
        Directory containing <model_name>.joblib files.
        RECOMMENDED:
            models/sktime/<label>/<frequency>/

    names : list of str
        Which model names to load.

    Returns
    -------
    models : dict
        {model_name: loaded_model}
    """
    folder = Path(folder)
    models = {}

    for name in names:
        path = folder / f"{name}.joblib"

        if not path.exists():
            raise FileNotFoundError(
                f"[model_io] ERROR: Expected model file not found:\n  {path}"
            )

        model = joblib.load(path)
        models[name] = model
        print(f"[model_io] Loaded {name} ← {path}")

    return models