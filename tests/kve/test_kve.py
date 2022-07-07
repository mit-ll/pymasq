from pymasq.kve.kve import key_variable_exploration
import pandas as pd
import pytest

from pymasq.kve import random_forest_scores, boruta_scores, rfe_scores, stepwise_scores
from pymasq.datasets import gen_num_df, gen_bin_df, load_census
from pymasq.preprocessing import EmbeddingsEncoder
from pymasq import ROOT_DIR


# Runs a smaller faster test, or uncomment line 13 for longer, larger test
params = [2000, 5000, 10000, 100000]
# params = [2000, 5000, 10000, 100000, 500000, 1000000]


@pytest.fixture
def my_df():
    return load_census()


@pytest.fixture(scope="session", params=params)
def bin_df(request):
    """
    Calls gen_bin_df with the parameterized dataset size
    """
    n = request.param
    return gen_bin_df(n)


@pytest.fixture(scope="session", params=params)
def num_df(request):
    """
    Calls gen_num_df with the parameterized dataset size
    """
    n = request.param
    return gen_num_df(n)


@pytest.fixture(scope="session", params=params)
def comb_df(request):
    """
    Generates a dataframe which is the combination of the binary and numeric dataframes
    defined in the bin_df and num_df functions so that we can test that the feature
    importance algorithms still work when the datatypes are combined.
    """
    n = request.param
    bin_df = gen_bin_df(n)
    num_df = gen_num_df(n)
    num_df.drop("Label", axis=1, inplace=True)
    df = pd.merge(bin_df, num_df, left_index=True, right_index=True)

    return df


def test_random_forest_cont(my_df):
    """
    Tests random_forest_scores if passed a continuous variable for y
    """
    sensitive_col = "age"
    my_df = EmbeddingsEncoder.encode(
        my_df,
        sensitive_col=sensitive_col,
        cache_location=ROOT_DIR + "/datasets/data/cache",
    )
    rf = random_forest_scores(
        x_train=my_df.drop(sensitive_col, axis=1),
        y=my_df[sensitive_col],
        verbose=0,
        categories=-1,
        n_estimators=20,
    )
    assert len(rf[1]) > 0, "Should be True"


def test_boruta_cont(my_df):
    """
    Tests boruta_scores if passed a continuous variable for y
    """
    sensitive_col = "age"
    my_df = EmbeddingsEncoder.encode(
        my_df,
        sensitive_col=sensitive_col,
        cache_location=ROOT_DIR + "/datasets/data/cache",
    )
    rf = boruta_scores(
        x_train=my_df.drop(sensitive_col, axis=1),
        y=my_df[sensitive_col],
        verbose=0,
        categories=-1,
        max_iter=5,
        n_estimators=20,
    )
    assert len(rf[1]) > 0, "Should be True"


def test_rfe_cont(my_df):
    """
    Tests rfe_scores if passed a continuous variable for y
    """
    sensitive_col = "age"
    my_df = EmbeddingsEncoder.encode(
        my_df,
        sensitive_col=sensitive_col,
        cache_location=ROOT_DIR + "/datasets/data/cache",
    )
    rf = rfe_scores(
        x_train=my_df.drop(sensitive_col, axis=1),
        y=my_df[sensitive_col],
        verbose=0,
        categories=-1,
    )
    assert len(rf[1]) > 0, "Should be True"


def test_random_forest_multiclass(my_df):
    """
    Tests random_forest_scores if passed a variable with number of categories > 2 for y
    """
    sensitive_col = "education"
    my_df = EmbeddingsEncoder.encode(
        my_df,
        sensitive_col=sensitive_col,
        cache_location=ROOT_DIR + "/datasets/data/cache",
    )
    y = my_df[sensitive_col]
    n_cats = len(y.dropna().unique())
    rf = random_forest_scores(
        x_train=my_df.drop(sensitive_col, axis=1),
        y=y,
        verbose=0,
        categories=n_cats,
        n_estimators=20,
    )
    assert len(rf[1]) > 0, "Should be True"


def test_boruta_multiclass(my_df):
    """
    Tests boruta_scores if passed a variable with number of categories > 2 for y
    """
    sensitive_col = "education"
    my_df = EmbeddingsEncoder.encode(
        my_df,
        sensitive_col=sensitive_col,
        cache_location=ROOT_DIR + "/datasets/data/cache",
    )
    y = my_df[sensitive_col]
    n_cats = len(y.dropna().unique())
    rf = boruta_scores(
        x_train=my_df.drop(sensitive_col, axis=1),
        y=y,
        verbose=0,
        categories=n_cats,
        max_iter=5,
        n_estimators=20,
    )
    assert len(rf[1]) > 0, "Should be True"


def test_rfe_multiclass(my_df):
    """
    Tests rfe_scores if passed a variable with number of categories > 2 for y
    """
    sensitive_col = "education"
    my_df = EmbeddingsEncoder.encode(
        my_df,
        sensitive_col=sensitive_col,
        cache_location=ROOT_DIR + "/datasets/data/cache",
    )
    y = my_df[sensitive_col]
    n_cats = len(y.dropna().unique())
    rf = rfe_scores(
        x_train=my_df.drop(sensitive_col, axis=1),
        y=y,
        verbose=2,
        categories=n_cats,
        cv=2,
    )
    assert len(rf[1]) > 0, "Should be True"


def test_random_forest_bin(bin_df):
    """
    Tests random_forest_scores feature importance ranks for a binary dataframe
    of a given size.
    """
    y = bin_df["Label"]
    n_cats = len(y.dropna().unique())
    rf = random_forest_scores(
        x_train=bin_df.drop("Label", axis=1), y=y, verbose=0, categories=n_cats
    )
    assert (
        rf[1] == ["yes"] + ["no"] * 5
    ), "Should be ['yes', 'no', 'no', 'no', 'no', 'no']]"


def test_boruta_bin(bin_df):
    """
    Tests boruta_scores feature importance ranks for a binary dataframe
    of a given size.
    """
    y = bin_df["Label"]
    n_cats = len(y.dropna().unique())
    assert boruta_scores(
        x_train=bin_df.drop("Label", axis=1), y=y, verbose=0, categories=n_cats
    ) == ["yes"] * 5 + [
        "maybe"
    ], "Should be ['yes', 'yes', 'yes', 'yes', 'yes', 'maybe']"


def test_rfe_bin(bin_df):
    """
    Tests rfe_scores feature importance ranks for a binary dataframe
    of a given size.
    """
    y = bin_df["Label"]
    n_cats = len(y.dropna().unique())
    assert (
        rfe_scores(
            x_train=bin_df.drop("Label", axis=1),
            y=y,
            verbose=0,
            categories=n_cats,
        )
        == ["yes"] + ["no"] * 5
    ), "Should be ['yes', 'no', 'no', 'no', 'no', 'no']"


def test_random_forest_num(num_df):
    """
    Tests random_forest_scores feature importance ranks for a numeric dataframe
    of a given size.
    """
    y = num_df["Label"]
    n_cats = len(y.dropna().unique())
    rf = random_forest_scores(
        x_train=num_df.drop("Label", axis=1), y=y, verbose=0, categories=n_cats
    )
    assert (
        rf[1] == ["yes"] + ["no"] * 5
    ), "Should be ['yes', 'no', 'no', 'no', 'no', 'no']]"


def test_boruta_num(num_df):
    """
    Tests boruta_scores feature importance ranks for a numeric dataframe
    of a given size.
    """
    y = num_df["Label"]
    n_cats = len(y.dropna().unique())
    assert (
        boruta_scores(
            x_train=num_df.drop("Label", axis=1),
            y=y,
            verbose=0,
            categories=n_cats,
        )
        == ["yes"] * 6
    ), "Should be ['yes', 'yes', 'yes', 'yes', 'yes', 'yes']"


def test_rfe_num(num_df):
    """
    Tests rfe_scores feature importance ranks for a numeric dataframe
    of a given size.
    """
    y = num_df["Label"]
    n_cats = len(y.dropna().unique())
    assert (
        rfe_scores(
            x_train=num_df.drop("Label", axis=1),
            y=y,
            verbose=0,
            categories=n_cats,
        )
        == ["yes"] + ["no"] * 5
    ), "Should be ['yes', 'no', 'no', 'no', 'no', 'no']"


def test_boruta_comb(comb_df):
    """
    Tests boruta_scores feature importance ranks for a combined dataframe
    of a given size.
    """
    if comb_df.shape[0] <= 2000:
        assert True
    y = comb_df["Label"]
    n_cats = len(y.dropna().unique())
    scores = boruta_scores(
        x_train=comb_df.drop("Label", axis=1), y=y, verbose=0, categories=n_cats
    )
    assert (
        scores == ["yes"] * 5 + ["maybe"] + ["yes"] * 6
    ), "One 'maybe' at index 5, otherwise all 'yes"


def test_random_forest_comb(comb_df):
    """
    Tests random_forest_scores feature importance ranks for a combined dataframe
    of a given size.
    """
    y = comb_df["Label"]
    n_cats = len(y.dropna().unique())
    scores = random_forest_scores(
        x_train=comb_df.drop("Label", axis=1),
        y=y,
        verbose=0,
        categories=n_cats,
    )
    assert (scores[1] == ["no"] * 6 + ["yes"] + ["no"] * 5) or (
        scores[1] == ["yes"] + ["no"] * 5 + ["yes"] + ["no"] * 5
    ), "Should follow one of two patterns"


def test_stepwise(my_df):
    """
    Tests stepwise feature importance ranks for census dataframe
    """
    sensitive_col = "sex"
    my_df = EmbeddingsEncoder.encode(
        my_df,
        sensitive_col=sensitive_col,
        cache_location=ROOT_DIR + "/datasets/data/cache",
    )
    scores = stepwise_scores(
        my_df.drop(sensitive_col, axis=1), my_df[sensitive_col], verbose=False
    )
    test_scores = [
        "no",
        "yes",
        "no",
        "no",
        "no",
        "yes",
        "yes",
        "no",
        "no",
        "no",
        "yes",
        "yes",
        "no",
        "no",
        "yes",
        "no",
        "no",
        "yes",
        "no",
        "no",
        "yes",
        "yes",
        "yes",
        "no",
        "yes",
        "yes",
        "yes",
        "yes",
        "yes",
        "yes",
        "yes",
        "no",
        "yes",
        "yes",
        "yes",
        "yes",
        "yes",
        "no",
        "no",
        "no",
        "no",
        "no",
        "no",
        "no",
        "no",
        "yes",
        "yes",
        "no",
        "no",
    ]
    assert scores == test_scores, "Should be True"


def test_kve_cont(my_df):
    """Tests if the entire kve process completes"""
    sensitive_col = "age"
    my_df = EmbeddingsEncoder.encode(
        my_df,
        sensitive_col=sensitive_col,
        cache_location=ROOT_DIR + "/datasets/data/cache",
    )
    kve = key_variable_exploration(
        my_df, sensitive_col=sensitive_col, verbose=False, preprocessed=True
    )
    assert len(kve) > 0, "Should be True"
