import numpy as np
import pandas as pd
import pytest

from pymasq.datasets import load_data, load_loan
from pymasq.mitigations import (
    s,  # shuffle.py module
    shuffle,
    MODEL,
)
from pymasq.errors import InputError, DataTypeError


@pytest.fixture
def my_df():
    df = load_data("prestige.csv").set_index("Unnamed: 0")
    df[["typeprof", "typewc"]] = pd.get_dummies(df.type, drop_first=True)
    df.drop("type", axis=1, inplace=True)
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
def loan_df():
    df = load_loan().dropna().reset_index()
    return df


def test_reverse_map():
    """
    Tests reverse_map function given set inputs that were used in the R
    sdcMicro library.
    """
    df = load_data("prestige.csv").set_index("Unnamed: 0")
    ystar = load_data("y_star.csv").set_index("Unnamed: 0")
    shuf = load_data("shuffled.csv").set_index("Unnamed: 0")
    assert shuf.equals(s._reverse_map(df, ystar)), "Should be True"


def test_shuffle_shuffle_cols_not_in_data(my_df):
    shuffle_cols = ["col-not-in-df", "other-col-not-in-df"]
    with pytest.raises(InputError):
        shuffle(my_df, shuffle_cols=shuffle_cols)


def test_shuffle_cor_cols_not_in_data(my_df):
    shuffle_cols = ["education"]
    cor_cols = ["col-not-in-df", "other-col-not-in-df"]
    with pytest.raises(InputError):
        shuffle(my_df, shuffle_cols=shuffle_cols, cor_cols=cor_cols)


def test_shuffle_overlapping_cols(my_df):
    shuffle_cols = ["education"]
    cor_cols = ["education", "income"]
    with pytest.raises(InputError):
        shuffle(my_df, shuffle_cols=shuffle_cols, cor_cols=cor_cols)


def test_shuffle_identical_values_in_cols(identical_df):
    shuffle_cols = ["a"]
    cor_cols = ["b", "c"]
    with pytest.raises(InputError):
        shuffle(identical_df, shuffle_cols=shuffle_cols, cor_cols=cor_cols)


def test_shuffle_cols_not_numeric(my_df):
    """
    Tests if error raised when input is not numeric
    """
    df = my_df[["income", "education", "women", "prestige", "typeprof", "typewc"]]
    df = df.astype({"women": str})
    with pytest.raises(InputError):
        shuffle(
            df,
            shuffle_cols=["women"],
            cor_cols=["education", "prestige", "typeprof", "typewc"],
        ).round(10)


def test_shuffle_same_mean_different_values(loan_df):
    """ Test that values are perturbed and retain the same mean while also in different order """
    shuffle_cols = ["ApplicantIncome", "LoanAmount"]
    cor_cols = ["Education", "Loan_Status"]
    shuffled = shuffle(
        loan_df,
        shuffle_cols=shuffle_cols,
        cor_cols=cor_cols,
    )
    assert all(
        loan_df[shuffle_cols].mean() == shuffled[shuffle_cols].mean()
    ), "Means are not equal"
    assert all(
        loan_df[shuffle_cols] != shuffled[shuffle_cols]
    ), "Shuffle returned same order"


def test_shuffle_returns_same_shapes(loan_df):
    shuffle_cols = ["ApplicantIncome", "LoanAmount"]
    cor_cols = ["Education", "Loan_Status"]

    in_size_1 = loan_df.shape
    out_size_1 = shuffle(
        loan_df,
        shuffle_cols=shuffle_cols,
        cor_cols=cor_cols,
    ).shape

    in_size_2 = loan_df[shuffle_cols + cor_cols].shape
    out_size_2 = shuffle(
        loan_df,
        shuffle_cols=shuffle_cols,
        cor_cols=cor_cols,
        cols=shuffle_cols + cor_cols,
    ).shape

    assert in_size_1 == out_size_1
    assert in_size_2 == out_size_2