#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pytest

from pymasq import config

config.FORMATTING_ON_OUTPUT = True

from pymasq import set_seed
from pymasq.datasets import load_loan
from pymasq.mitigations import microaggregation as magg
from pymasq.mitigations.microaggregation import MaggMethods
from pymasq.errors import InputError, LessThanOrEqualToZeroError, NotInRangeError


METHODS = [
    MaggMethods.ADVANCED,
    MaggMethods.QUANTILE,
    MaggMethods.RANKING,
    MaggMethods.ROBUST,
    MaggMethods.SEQUENTIAL,
]
MAGG_ADVANCED_KWARGS = {"clust": "kmeans", "scale": "standard", "reduct": "pca"}
NUM_RECORDS = 100
DELTA = int(NUM_RECORDS * 0.10)

set_seed(1337)


@pytest.fixture
def rand_df():
    return pd.DataFrame(np.random.randint(1, NUM_RECORDS, (NUM_RECORDS, 4)))


@pytest.fixture
def my_df():
    df = load_loan()
    cols = ["LoanAmount", "ApplicantIncome", "CoapplicantIncome"]
    df = df.dropna().reset_index(drop=True).loc[:NUM_RECORDS, cols]
    return df


def test_magg_error_if_invalid_method(my_df):
    """ Test that microaggregation throws an InputError if incorrect method is supplied. """
    with pytest.raises(InputError):
        magg(my_df, method=None, aggr=2)


@pytest.mark.parametrize("method", METHODS)
def test_magg_returns_same_dimensions_and_column_names(my_df, method):
    """ Test that microaggregation returns the same dimensions and column names. """
    kwargs = {}
    if method == MaggMethods.ADVANCED:
        kwargs = MAGG_ADVANCED_KWARGS
    test_df = magg(my_df, method=method, aggr=2, **kwargs)
    assert test_df.shape == my_df.shape
    assert all(test_df.columns == my_df.columns)


@pytest.mark.parametrize("method", (METHODS))
def test_magg_aggr_is_valid(my_df, method):
    """ Test for NotInRangeError when `aggr` not in [1, len(my_df)] """
    aggr = 0
    kwargs = {}
    if method == MaggMethods.ADVANCED:
        kwargs = MAGG_ADVANCED_KWARGS
    if method == "quantile":
        with pytest.raises(NotInRangeError):
            magg(my_df, method=method, aggr=aggr, **kwargs)
    else:
        with pytest.raises(LessThanOrEqualToZeroError):
            magg(my_df, method=method, aggr=aggr, **kwargs)


@pytest.mark.parametrize("method", (METHODS))
def test_magg_unique_vals_is_one(my_df, method):
    """ Test the number of unique values returned for `aggr` is 1. """
    kwargs = {}
    if method == MaggMethods.ADVANCED:
        kwargs = MAGG_ADVANCED_KWARGS
    aggr = 1
    test_df = magg(
        my_df, method=method, aggr=aggr, keep_dtypes=True, **kwargs
    )  # .astype(int)
    assert True == np.allclose(my_df, test_df, 1, 1)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("aggr", [2] + [n for n in range(10, NUM_RECORDS, 10)])
def test_magg_unique_vals_greater_than_one(my_df, method, aggr):
    """ Test the number of unique values returned for `aggr` is greater than 1. """
    kwargs = {}
    if method == MaggMethods.ADVANCED:
        kwargs = MAGG_ADVANCED_KWARGS
    test_df = magg(my_df, method=method, aggr=aggr, keep_dtypes=True, **kwargs)
    nunique = len(test_df) // aggr
    assert all([val <= (nunique + 1) for val in test_df.nunique()])


def test_magg_quantile_not_in_range(my_df):
    """ Test that quantile-based microaggregation throws an NotInRangeError if aggr > len(my_df). """
    with pytest.raises(NotInRangeError):
        magg(my_df, method="quantile", aggr=len(my_df) + 1)


def test_magg_advanced_required_extra_parameters(my_df):
    """ Test that advanced-based microaggregation throws an InputError if neither clust or reduct are specified. """
    with pytest.raises(InputError):
        magg(my_df, method="advanced")


def test_magg_advanced_error_if_invalid_methods(my_df):
    """ Test that advanced-based microaggregation throws an InputError when input kwargs are not valid. """
    with pytest.raises(InputError):
        magg(my_df.copy(), method="advanced", clust="INVALID")
        magg(my_df.copy(), method="advanced", reduct="INVALID")
        magg(my_df.copy(), method="advanced", scale="INVALID", reduct="INVALID")
