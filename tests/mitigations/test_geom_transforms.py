import numpy as np
import pandas as pd
import pytest

from pymasq.config import DEFAULT_SEED
from pymasq.mitigations import geom_transform
from pymasq.errors import InputError

rg = np.random.default_rng(DEFAULT_SEED)

@pytest.fixture
def my_rand_df():
    ncols = 5
    colnames = "abcdefghijklmnopqrstuvwxyz"
    df = pd.DataFrame(
        rg.integers(0, 100, (100, ncols)),
        columns=[colnames[i] for i in range(ncols)],
    )
    return df


@pytest.fixture
def identical_df():
    ncols = 3
    colnames = "abcdefghijklmnopqrstuvwxyz"
    df = pd.DataFrame(
        np.ones(shape=(100, ncols)), columns=[colnames[i] for i in range(ncols)]
    )
    return df


@pytest.fixture
def my_non_numeric_df():
    ncols = 3
    colnames = list("abcdefghijklmnopqrstuvwxyz")
    df = pd.DataFrame(
        rg.choice(colnames, size=(100, ncols), replace=True),
        columns=colnames[:ncols],
    )
    return df


def test_geom_transform_perturb_cols_not_in_data(my_rand_df):
    perturb_cols = ["col-not-in-df"]
    with pytest.raises(InputError):
        geom_transform(my_rand_df, perturb_cols=perturb_cols)


def test_geom_transform_sensitive_col_not_in_data(my_rand_df):
    perturb_cols = ["education"]
    sensitive_col = "col-not-in-df"
    with pytest.raises(InputError):
        geom_transform(
            my_rand_df, perturb_cols=perturb_cols, sensitive_col=sensitive_col
        )


def test_geom_transform_overlapping_cols(my_rand_df):
    perturb_cols = ["education"]
    sensitive_col = "education"
    with pytest.raises(InputError):
        geom_transform(
            my_rand_df, perturb_cols=perturb_cols, sensitive_col=sensitive_col
        )


def test_geom_transform_identical_values_in_cols(identical_df):
    perturb_cols = ["a"]
    sensitive_col = "b"
    with pytest.raises(InputError):
        geom_transform(
            identical_df, perturb_cols=perturb_cols, sensitive_col=sensitive_col
        )


def test_geom_transform_perturb_cols_not_numeric(my_non_numeric_df):
    """
    Tests if error raised when input is not numeric
    """
    with pytest.raises(InputError):
        geom_transform(
            my_non_numeric_df,
            perturb_cols=["a", "b"],
            sensitive_col="c",
        )


def test_geom_transform_error_single_column(my_rand_df):
    perturb_cols = ["a"]
    sensitive_col = "c"
    with pytest.raises(InputError):
        geom_transform(
            my_rand_df,
            perturb_cols=perturb_cols,
            sensitive_col=sensitive_col,
        )


def test_geom_transform_different_values_for_perturb_cols(my_rand_df):
    """Ensure geom_transform returns different values for perturb_cols"""
    perturb_cols = ["a", "b"]
    sensitive_col = "c"
    rdf = geom_transform(
        my_rand_df,
        perturb_cols=perturb_cols,
        sensitive_col=sensitive_col,
    )
    assert all(
        my_rand_df[perturb_cols] != rdf[perturb_cols]
    ), "Geom Transform returned same input values"


def test_geom_transform_cols_not_specified_no_perturbed(my_rand_df):
    """Ensure geom_transform returns different values for perturb_cols"""
    perturb_cols = ["a", "b"]
    sensitive_col = "d"
    ignore_cols = ["c"]
    rdf = geom_transform(
        my_rand_df,
        perturb_cols=perturb_cols,
        sensitive_col=sensitive_col,
    )
    assert all(
        my_rand_df[ignore_cols] == rdf[ignore_cols]
    ), "Geom Transform did not return same input values for cols not specified with shuffling"

    rdf = geom_transform(
        my_rand_df,
        perturb_cols=perturb_cols,
        sensitive_col=sensitive_col,
        shuffle=False,
    )
    assert all(
        my_rand_df[ignore_cols] == rdf[ignore_cols]
    ), "Geom Transform did not return same input values for cols not specified with NO shuffling"


def test_geom_transform_same_values_for_sensitive_col(my_rand_df):
    """Ensure geom_transform returns different values for perturb_cols"""
    perturb_cols = ["a", "b"]
    sensitive_col = "c"
    rdf = geom_transform(
        my_rand_df,
        perturb_cols=perturb_cols,
        sensitive_col=sensitive_col,
    )
    assert (
        my_rand_df[sensitive_col].values.squeeze().sort()
        == rdf[sensitive_col].values.ravel().sort()
    ), "Geom Transform did not return the same input values for sensitive_col with shuffling"


def test_geom_transform_same_values_in_proper_order_for_sensitive_col(my_rand_df):
    """Ensure geom_transform returns different values for perturb_cols"""
    perturb_cols = ["a", "b"]
    sensitive_col = "c"
    rdf = geom_transform(
        my_rand_df,
        perturb_cols=perturb_cols,
        sensitive_col=sensitive_col,
        shuffle=False,
    )
    assert all(
        my_rand_df[sensitive_col].values.ravel() == rdf[sensitive_col].values.ravel()
    ), "Geom Transform did not return the same input values for sensitive_col with NO shuffling"


def test_geom_transform_returns_same_shapes(my_rand_df):
    """Ensure geom_transform returns the same dataframe shapes"""
    perturb_cols = ["a", "b"]
    sensitive_col = "d"

    in_size_1 = my_rand_df.shape
    out_size_1 = geom_transform(
        my_rand_df,
        perturb_cols=perturb_cols,
        sensitive_col=sensitive_col,
    ).shape

    in_size_2 = my_rand_df[perturb_cols + [sensitive_col]].shape
    out_size_2 = geom_transform(
        my_rand_df,
        perturb_cols=perturb_cols,
        sensitive_col=sensitive_col,
        cols=perturb_cols + [sensitive_col],
    ).shape

    assert in_size_1 == out_size_1
    assert in_size_2 == out_size_2
