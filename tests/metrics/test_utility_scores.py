from random import sample, gauss, seed

import pandas as pd
import pytest
import scipy.stats as ss

from pymasq.datasets import load_census
from pymasq.metrics import propensity_score, proportion_diff_score, jensen_shannon

# These tests may fail for fewer rows than listed below
params = [5000, 10000, 100000]
seed = 1234


@pytest.fixture(scope="session", params=params)
def orig_bin_df(request):
    """
    Generates a pandas dataframe that has a column and label with matching
    binary values of 0.
    """
    n = request.param
    orig_dict = {"Col": [0] * n, "Label": [0] * n}
    return pd.DataFrame(orig_dict)


@pytest.fixture(scope="session", params=params)
def orig_cat_df(request):
    """
    Generates a pandas dataframe that has a column and label with matching
    binary values of 0.
    """
    n = request.param
    orig_dict = {"Col": ["A"] * n, "Label": [0] * n}
    return pd.DataFrame(orig_dict)


@pytest.fixture(scope="session", params=params)
def orig_num_df(request):
    """
    Generates a pandas dataframe that has a column and label, where the
    column all derived from a N(5, .1) distribution and the label is
    the binary class 0.
    """
    n = request.param
    orig_dict = {"Col": [gauss(5, 0.1) for _ in range(n)], "Label": [0] * n}
    return pd.DataFrame(orig_dict)


@pytest.fixture
def my_df():
    df = load_census()
    return df


import time


#################### Tests of Propensity ElasticNet ############################
def test_propensity_bin(orig_bin_df, orig_cat_df):
    """
    Tests propensity_score for binary data that increasingly changes testing
    that scores increases from 0 towards 1 as more of the data changes.
    """
    for classifier, pp in [
        ("logreg", ["embeddings", "label_encode"]),
        ("rfclass", ["embeddings", "label_encode"]),
        ("js", ["embeddings", "label_encode"]),
        ("prop_diff", ["None"]),
    ]:
        for preprocessor in pp:
            for orig_df, mod_val in [
                (orig_bin_df, 1),  # numerical column
                (orig_cat_df, "B"),  # categorical column
            ]:
                scores = []
                ratios = [i / 100 for i in range(0, 110, 10)]
                n = orig_df.shape[0]
                orig_df.Col.loc[0] = mod_val
                for r in ratios:
                    elements = sample(range(n), int(n * r))
                    mod_df = orig_df.copy()
                    mod_df["Label"] = 1
                    mod_df.loc[elements, "Col"] = mod_val
                    start = time.perf_counter_ns()
                    if classifier == "prop_diff":
                        score = proportion_diff_score(
                            orig_df=orig_df.drop(["Label"], axis=1),
                            mod_df=mod_df.drop(["Label"], axis=1),
                        )
                    elif classifier == "js":
                        score = jensen_shannon(
                            orig_df=orig_df,
                            mod_df=mod_df,
                            sensitive_col="Label",
                        )
                    else:
                        score = propensity_score(
                            orig_df=orig_df,
                            mod_df=mod_df,
                            sensitive_col="Label",
                            method=classifier,
                            preprocessor=preprocessor,
                        )
                    stop = time.perf_counter_ns()
                    type = "cat" if mod_val == "B" else "num"
                    scores.append(score)
                ranks = list(ss.rankdata([i for i in scores], method="dense"))
                res = [i for i in range(1, len(ratios) + 1)]
                assert (
                    ranks == res
                ), f"{classifier}/{preprocessor}: Should be {res} but got {ranks} with {scores}"
                assert scores[-1] >= 0.99


def test_propensity_score_identical(my_df):
    """
    Tests propensity_score for identical data frames
    """
    print()
    for classifier, pp in [
        ("logreg", ["embeddings", "label_encode"]),
        ("rfclass", ["embeddings", "label_encode"]),
        ("js", ["embeddings", "label_encode"]),
        ("prop_diff", ["None"]),
    ]:
        for preprocessor in pp:
            if classifier == "prop_diff":
                score = proportion_diff_score(
                    orig_df=my_df.drop(["sex"], axis=1),
                    mod_df=my_df.drop(["sex"], axis=1),
                )
            elif classifier == "js":
                score = jensen_shannon(
                    orig_df=my_df,
                    mod_df=my_df,
                    sensitive_col="sex",
                )
            else:
                score = propensity_score(
                    orig_df=my_df,
                    mod_df=my_df,
                    sensitive_col="sex",
                    method=classifier,
                    preprocessor=preprocessor,
                )
            print(f"{classifier}/{preprocessor}: {round(score,2)}")
            assert (
                round(score, 2) <= 0.0
            ), f"{classifier}/{preprocessor}: Should be 0.0 but is round({score},2)={round(score,2)}"


def test_propensity_score_moderate_change(my_df):
    """
    Tests propensity_score for a moderate change to the original data
    """
    expected = [0.72, 0.01, 0.97, 0.0, 0.0, 0.0, 0.13]
    for classifier, pp in [
        ("logreg", ["embeddings", "label_encode"]),
        ("rfclass", ["embeddings", "label_encode"]),
        ("js", ["embeddings", "label_encode"]),
        ("prop_diff", ["None"]),
    ]:
        for preprocessor in pp:
            my_df2 = my_df.copy()
            my_df2["age"] = my_df["age"].sample(frac=1, random_state=1234).values
            my_df2["education"] = (
                my_df["education"].sample(frac=1, random_state=1234).values
            )
            if classifier == "prop_diff":
                score = proportion_diff_score(
                    orig_df=my_df.drop(["sex"], axis=1),
                    mod_df=my_df2.drop(["sex"], axis=1),
                )
            elif classifier == "js":
                score = jensen_shannon(
                    orig_df=my_df,
                    mod_df=my_df2,
                    sensitive_col="sex",
                )
            else:
                score = propensity_score(
                    orig_df=my_df,
                    mod_df=my_df2,
                    sensitive_col="sex",
                    method=classifier,
                    preprocessor=preprocessor,
                )
            print(f"{classifier}/{preprocessor}: {round(score,2)}")
            exp = expected.pop()
            assert (
                round(score, 2) == exp,
            ), f"Should be {exp} but is round({score},2)={round(score,2)}"
