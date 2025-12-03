from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.deep_learning import InceptionTimeClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier


def get_sktime_models(model_num: int = -1):
    """
    Return a dictionary of initialized sktime TSC models.

    Parameters
    ----------
    model_num : int
        -1  -> return ALL models (default)
         1  -> return ONLY the first model
         2  -> return ONLY the second model
         3  -> etc...
        (Order is defined by the dictionary below.)

    Notes
    -----
    - Commented models remain untouched. If you uncomment them later,
      they automatically become selectable via model_num.
    """
    # -----------------------------
    # Define available models
    # -----------------------------
    models = {
        # 1) Bake-off baseline: 1NN Euclidean
        "1NN_ED": KNeighborsTimeSeriesClassifier(
            n_neighbors=1,
            distance="euclidean",
            algorithm= "ball_tree", #"brute_incr", #"brute" or
            n_jobs=-1
        ),

        # 2) ROCKET (currently commented out)
         "ROCKET": RocketClassifier(
             num_kernels=int(512 * 1.4),
             n_jobs=-1,
             random_state=42,
         ),

        # 3) InceptionTime (currently commented out)
         "InceptionTime": InceptionTimeClassifier(
             n_epochs=2,
             batch_size=64,
             random_state=42,
             activation_inception="relu",
             verbose=False,
         ),

         #4) HIVE-COTE (currently commented out)
     #    "HIVECOTEV2": HIVECOTEV2(
      #       time_limit_in_minutes=1.0,
       #      n_jobs=-1,
       #      random_state=42,
       #      verbose=0,
        # ),
    }

    # -----------------------------
    # Model selection logic
    # -----------------------------
    if model_num == -1:
        return models

    keys = list(models.keys())

    if model_num < 1 or model_num > len(keys):
        raise ValueError(
            f"Requested model_num={model_num}, "
            f"but only {len(keys)} model(s) available. "
            f"Available indices: 1..{len(keys)}."
        )

    selected_key = keys[model_num - 1]
    return {selected_key: models[selected_key]}