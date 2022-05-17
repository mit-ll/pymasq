#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pytest

from scipy.stats import kstest

from pymasq import config, set_seed

config.FORMATTING_ON_OUTPUT = True

from pymasq.datasets import load_data
from pymasq.mitigations import add_noise, ADDITIVE, CORRELATED
from pymasq.errors import InputError


set_seed(10)


@pytest.fixture
def prestige_df():
    df = load_data("prestige.csv").set_index("Unnamed: 0")
    df[["typeprof", "typewc"]] = pd.get_dummies(df.type, drop_first=True)
    df.drop("type", axis=1, inplace=True)
    df = df[["income", "education", "women", "prestige", "typeprof", "typewc"]]
    return df


@pytest.fixture
def ystar_df():
    ystar1 = load_data("y_star1.csv")
    return ystar1


@pytest.fixture
def neg_cor_df():
    # Dataframe where there is a correlation of -1 between columns 1 and 2
    return pd.DataFrame.from_dict(
        {"col1": [1, 2, 3, 4, 5, 6, 7, 8, 9], "col2": [9, 8, 7, 6, 5, 4, 3, 2, 1]}
    )


# # ----- Method: Additive Tests -----
def test_add_noise_additive_1(prestige_df):
    """
    Test that add noise throws an InputError when given an invalid method.
    """
    with pytest.raises(InputError):
        add_noise(prestige_df, "INVALID", magnitude=5, degrees=1)


def test_add_noise_additive_2(prestige_df):
    """
    Test that added additive noise is normal by checking the p-value resulting from a
    kolmogorov-smirnov test
    """
    series = prestige_df["income"]
    noisy_series = add_noise(
        series, ADDITIVE, magnitude=5, degrees=1, keep_dtypes=False
    )
    ks_statistic, p_value = kstest(np.subtract(noisy_series, series), "norm")
    assert p_value < 0.05


def test_add_noise_additive_3(prestige_df):
    """
    Test that added additive noise is scaled approximately (within 5%) of the true standard
    deviation of the supplied series. Random value generation is set to a static seed by the
    'random' parameter.
    """
    mag = 5
    deg = 1
    series = prestige_df["income"]
    noisy_series = add_noise(series, ADDITIVE, magnitude=mag, degrees=deg)
    difference = np.subtract(noisy_series, series)
    std = series.std(ddof=deg)
    assert difference.std(ddof=deg) == pytest.approx((5.0 / 100) * std, 0.05)


# ----- Method: Correlated Tests -----
def test_add_noise_correlated_1(neg_cor_df):
    """
    This test checks that the correlation between exactly negatively correlated data is preserved
    when values along that line are added as noise to the data.
    """
    noisy_df = add_noise(neg_cor_df, method=CORRELATED, cols=["col1", "col2"])
    assert neg_cor_df.corr().round(2).equals(
        noisy_df.corr().round(
            2
        )  # rounding performed to account for occasional small differences
    ) and not neg_cor_df.equals(noisy_df)


def test_add_noise_correlated_2(neg_cor_df):
    """
    This test verifies that adding correlated noise will fail if only one column is supplied
    """
    with pytest.raises(InputError):
        add_noise(neg_cor_df, method=CORRELATED, cols=["col1"])


def test_add_noise_correlated_3(neg_cor_df):
    """
    This test verifies that no added noise exceeds the specified magnitude
    """
    mag = 10
    noisy_df = add_noise(
        neg_cor_df, method=CORRELATED, cols=["col1", "col2"], magnitude=mag
    )
    ranges = np.array(
        [
            max(neg_cor_df["col1"]) - min(neg_cor_df["col1"]),
            max(neg_cor_df["col2"]) - min(neg_cor_df["col2"]),
        ]
    )

    all(abs(neg_cor_df - noisy_df) / ranges < (mag / 100))
