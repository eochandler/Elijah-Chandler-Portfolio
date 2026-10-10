from __future__ import annotations

import numpy as np

from src.turbofan.data import synthetic_fixture


def test_loader_has_expected_columns_and_no_missing_values():
    train_df, _, _, _ = synthetic_fixture()
    assert len(train_df.columns) >= 26
    assert train_df['rul'].notna().all()
    assert (train_df['rul'] >= 0).all()


def test_rul_is_clipped_and_correct():
    train_df, _, _, _ = synthetic_fixture()
    expected = train_df.groupby('unit')['cycle'].transform('max') - train_df['cycle']
    assert np.all(train_df['rul'] <= 125)
    assert np.allclose(train_df['rul'], expected.clip(upper=125))
