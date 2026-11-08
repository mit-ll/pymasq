#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pytest

from pymasq import config

config.FORMATTING_ON_OUTPUT = True

from pymasq.errors import InputError, NotInRangeError
from pymasq.mitigations import rank_swap


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "a": np.arange(100, dtype=float),
        "b": np.random.normal(50, 10, 100),
        "c": np.random.uniform(0, 1, 100),
    })


def test_rank_swap_shape_and_columns(sample_df):
    res = rank_swap(sample_df, p=0.1, seed=123)
    assert res.shape == sample_df.shape
    assert list(res.columns) == list(sample_df.columns)


def test_rank_swap_preserves_multiset(sample_df):
    res = rank_swap(sample_df, p=0.15, seed=123)
    for col in sample_df.columns:
        np.testing.assert_allclose(
            np.sort(res[col].to_numpy()),
            np.sort(sample_df[col].to_numpy()),
        )


def test_rank_swap_preserves_mean_and_sum(sample_df):
    res = rank_swap(sample_df, p=0.2, seed=42)
    for col in sample_df.columns:
        assert np.isclose(res[col].sum(), sample_df[col].sum())
        assert np.isclose(res[col].mean(), sample_df[col].mean())


def test_rank_swap_actually_swaps(sample_df):
    res = rank_swap(sample_df, p=0.2, seed=42)
    # At least some values should be at different positions
    assert not res["a"].equals(sample_df["a"])


def test_rank_swap_percentage_scale(sample_df):
    # p=10 (meaning 10%) should behave like p=0.10
    res_pct = rank_swap(sample_df, p=10, seed=123)
    res_frac = rank_swap(sample_df, p=0.10, seed=123)
    pd.testing.assert_frame_equal(res_pct, res_frac)


def test_rank_swap_reproducibility(sample_df):
    res1 = rank_swap(sample_df, p=0.1, seed=999)
    res2 = rank_swap(sample_df, p=0.1, seed=999)
    pd.testing.assert_frame_equal(res1, res2)


def test_rank_swap_subset_cols(sample_df):
    res = rank_swap(sample_df, p=0.1, cols=["a"], seed=42)
    assert "a" in res.columns
    # Multi-set of 'a' preserved
    np.testing.assert_allclose(
        np.sort(res["a"].to_numpy()),
        np.sort(sample_df["a"].to_numpy()),
    )


def test_rank_swap_with_nans():
    df = pd.DataFrame({
        "val": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0]
    })
    res = rank_swap(df, p=0.3, seed=42)
    # NaN positions should remain NaNs
    assert res["val"].isna().tolist() == df["val"].isna().tolist()
    # Non-NaN values should be a permutation of original non-NaN values
    valid_orig = df["val"].dropna().to_numpy()
    valid_res = res["val"].dropna().to_numpy()
    np.testing.assert_allclose(np.sort(valid_res), np.sort(valid_orig))


def test_rank_swap_identical_values():
    df = pd.DataFrame({"col": [5.0] * 20})
    res = rank_swap(df, p=0.2, seed=42)
    pd.testing.assert_frame_equal(res, df)


def test_rank_swap_series_input():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    res = rank_swap(s, p=0.2, seed=42)
    assert isinstance(res, pd.Series)
    np.testing.assert_allclose(np.sort(res.to_numpy()), np.sort(s.to_numpy()))


def test_rank_swap_invalid_p(sample_df):
    with pytest.raises(NotInRangeError):
        rank_swap(sample_df, p=0)
    with pytest.raises(NotInRangeError):
        rank_swap(sample_df, p=-0.5)
    with pytest.raises(NotInRangeError):
        rank_swap(sample_df, p=105)


def test_rank_swap_invalid_cols(sample_df):
    with pytest.raises(InputError):
        rank_swap(sample_df, cols=["non_existent_column"])


def test_rank_swap_non_numeric():
    df = pd.DataFrame({"text": ["a", "b", "c", "d"]})
    with pytest.raises(InputError):
        rank_swap(df)
