import logging
import numpy as np
import pandas as pd
import pytest

from pymasq.config import DEFAULT_SEED
from pymasq.datasets import load_census
from pymasq.errors import InputError, NotInRangeError
from pymasq.mitigations import pram

logger = logging.getLogger(__name__)

rg = np.random.default_rng(DEFAULT_SEED)

@pytest.fixture
def my_df():
    df = load_census()
    df = df[["workclass", "education", "relationship", "race", "sex"]].head(10)
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
def my_numerical_df():
    ncols = 10
    nrows = 10
    max_val = 1000000
    return pd.DataFrame(
        rg.integers(0, max_val, (nrows, ncols)),
        columns=[f"c{i}" for i in range(ncols)],
    )


def test_pram_perturb_cols_not_in_data(my_df):
    perturb_cols = ["col-not-in-df", "other-col-not-in-df"]
    with pytest.raises(InputError):
        pram(my_df, perturb_cols=perturb_cols)


def test_pram_sensitive_col_not_in_data(my_df):
    sensitive_col = "col-not-in-df"
    with pytest.raises(InputError):
        pram(my_df, sensitive_col=sensitive_col)


def test_pram_overlapping_cols(my_df):
    perturb_cols = ["workclass", "education"]
    sensitive_col = "education"
    with pytest.raises(InputError):
        pram(my_df, perturb_cols=perturb_cols, sensitive_col=sensitive_col)


def test_pram_identical_values_in_cols(identical_df):
    perturb_cols = ["b", "c"]
    sensitive_col = "a"
    with pytest.raises(InputError):
        pram(identical_df, perturb_cols=perturb_cols, sensitive_col=sensitive_col)


def test_pram_alpha_in_interval(my_df):
    with pytest.raises(NotInRangeError):
        pram(my_df, alpha=1.1)
    with pytest.raises(NotInRangeError):
        pram(my_df, alpha=-0.1)


def test_pram_probs_in_interval(my_df):
    with pytest.raises(NotInRangeError):
        pram(my_df, probs=1.1)
    with pytest.raises(NotInRangeError):
        pram(my_df, probs=-0.1)


def test_pram_probs_invalid_dict(my_df):
    probs = dict(
        key_not_in_my_df=pd.DataFrame(
            {"White": 0.5, "Black": 0.5}, index=["White", "Black"]
        )
    )
    with pytest.raises(InputError):
        pram(my_df, perturb_cols=["race"], probs=probs)
    with pytest.raises(InputError):
        pram(my_df, probs=probs)


def test_pram_probs_valid_dict(my_df):
    """Ensure that specifying probabilities results in that number of changes on average"""
    probs = dict(
        race=pd.DataFrame({"White": 0.5, "Black": 0.5}, index=["White", "Black"])
    )
    nrows = len(my_df)
    trials = 100
    counts = {"White": 0, "Black": 0}
    for _ in range(trials):
        r = pram(my_df, perturb_cols=["race"], probs=probs)
        c = r["race"].value_counts()
        counts["White"] += c["White"]
        counts["Black"] += c["Black"]

    threshold = 15
    assert abs(counts["Black"] - counts["White"]) < threshold
    assert abs((nrows * trials) - (counts["Black"] + counts["White"])) < threshold


def test_pram_numerical_cast_to_categorical(my_numerical_df):
    try:
        pram(my_numerical_df)
    except Exception as e:
        logger.exception(e)
        assert False, "Numerical dataframe should not have raised error."


def test_pram_returns_same_shapes(my_df):
    in_size = my_df.shape
    out_size = pram(my_df).shape
    assert in_size == out_size, f"Invalid shapes {in_size} vs {out_size}"


def test_pram_probs_equal_0(my_df):
    """at least 1 value changed"""
    r = pram(my_df, probs=0)
    assert not all((r == my_df).all())


def test_pram_probs_equal_1(my_df):
    """no change in data"""
    r = pram(my_df, probs=1)
    assert all((r == my_df).all())


def test_pram_alpha_equal_0(my_df):
    """no change in data"""
    r = pram(my_df, alpha=0)
    assert all((r == my_df).all())


def test_pram_alpha_equal_1(my_df):
    """at least 1 value changed"""
    r = pram(my_df, alpha=1)
    assert not all((r == my_df).all())


def test_pram_changes_cats(my_df):
    r = pram(my_df)
    assert not all((r == my_df).all()), "DataFrames should not be equal"


def test_pram_diff_results_w_sensitive_col_specified(my_df):
    r1 = pram(my_df)
    r2 = pram(my_df, sensitive_col="sex")

    assert not r1.equals(r2), "DataFrames should not be equal"
    assert np.all(r2["sex"] == my_df["sex"]), "sensitive_col should be equal"
