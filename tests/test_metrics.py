from __future__ import annotations

import numpy as np

from src.turbofan.metrics import nasa_score, rmse


def test_rmse_matches_hand_calculation():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 5.0])
    assert abs(rmse(y_true, y_pred) - np.sqrt(5.0 / 3.0)) < 1e-9


def test_nasa_score_has_late_penalty():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([5.0, 30.0])
    assert nasa_score(y_true, y_pred) > 0
    assert nasa_score(y_true, y_true) == 0.0
