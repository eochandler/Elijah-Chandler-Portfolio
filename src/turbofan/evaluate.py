from __future__ import annotations

import numpy as np

from .metrics import nasa_score, rmse


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Compute RMSE and NASA score for arrays of predictions."""
    return rmse(y_true, y_pred), nasa_score(y_true, y_pred)
