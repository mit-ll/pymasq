import pytest

from pymasq.datasets import load_census
from pymasq.metrics import (
    is_l_diverse,
    l_diversity,
    is_t_close,
    t_closeness,
    is_beta_like,
    beta_likeness,
    k_anon,
    auc_score,
)
from pymasq.datasets.data_generator import (
    generate_l_diverse_table,
    generate_t_close_table,
)
from pymasq.errors import InputError


@pytest.fixture
def my_df():
    df = load_census()
    cols = ["age", "education", "marital_status", "sex", "capital_gain"]
    df = df.loc[:100, cols]
    return df


LETTER_SET = ["A", "B", "C", "A", "B", "A", "C", "C", "B"]

true_assert_statement: str = "Should be True"

@pytest.fixture
def letter_df():
    return generate_t_close_table(LETTER_SET)


def test_l_diversity_all_same():
    """
    Tests l-diversity function
    """
    df = generate_l_diverse_table(2)
    assert l_diversity(df, "sensitive", 3) == pytest.approx(1.0), true_assert_statement


def test_l_diversity_variety():
    """
    Tests l-diversity function
    """
    df = generate_l_diverse_table([2, 3, 3, 2, 2])
    assert l_diversity(df, "sensitive", 2) == pytest.approx(0.6), true_assert_statement


def test_t_closeness_num():
    """
    Tests t-closeness on a dataset founnd in the paper "t-Closeness: Privacy Beyond k-Anonymity and l-Diversity"
    """
    tc_table = generate_t_close_table([3, 4, 5, 6, 8, 11, 7, 9, 10])
    expected_result = [
        0.37500000000000006,
        0.37500000000000006,
        0.37500000000000006,
        0.16666666666666663,
        0.16666666666666663,
        0.16666666666666663,
        0.23611111111111113,
        0.23611111111111113,
        0.23611111111111113,
    ]

    assert (
        t_closeness(tc_table, "sensitive", test=True, datatype="numeric")
        == expected_result
    ), true_assert_statement


def test_t_closeness_cat():
    """
    Tests t-closeness on a toy dataset
    """
    tc_table = generate_t_close_table(LETTER_SET)
    expected_result = [
        0.0,
        0.0,
        0.0,
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
    ]

    assert (
        t_closeness(tc_table, "sensitive", test=True, datatype="categorical")
        == expected_result
    ), true_assert_statement


def test_t_closeness():
    """
    Tests t-closeness on a toy dataset for overall comparison
    """
    tc_table = generate_t_close_table(LETTER_SET)
    assert (
        t_closeness(tc_table, "sensitive", datatype="categorical", t=0.0) == pytest.approx(1.0)
    ), true_assert_statement


def test_beta_likeness_1(letter_df):
    """
    Tests a fail condition for beta-likeness (beta value <= 0)
    """
    with pytest.raises(InputError):
        beta_likeness(letter_df, "sensitive", beta=0)


def test_beta_likeness_2(letter_df):
    """
    Tests beta-likeness on a toy dataset with a very small beta (any information gain should fail)
    """
    assert (
        beta_likeness(letter_df, "sensitive", beta=1e-9) == pytest.approx(4.0 / 9.0)
    ), "Should fail beta likeness on the 2 A's in EC2 and 2 C's in EC3"


def test_beta_likeness_3(letter_df):
    """
    Tests that the is_beta_like function returns correctly for a toy dataset with a very small
    beta and a very large beta
    """
    assert not is_beta_like(
        letter_df, "sensitive", beta=1e-9, enhanced=False
    ) and is_beta_like(
        letter_df, "sensitive", beta=1e9, enhanced=False
    ), "Should return False for the first function and True for the second"


METHODS = ["logreg", "tpotclass", "rfclass"]
PREPROCESSORS = ["embeddings", "label_encode"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
def test_auc_score_1(my_df, method, preprocessor):
    """
    Tests auc_drop on a toy dataset for overall comparison
    """
    kwargs = {"verbose": 1}
    if method == "tpot":
        kwargs = {"generations": 5, "population_size": 10}
    assert (
        auc_score(
            orig_df=my_df,
            mod_df=my_df.copy(),
            sensitive_col="sex",
            method=method,
            modeling_task="binary",
            preprocessor=preprocessor,
            **kwargs,
        )
        == pytest.approx(1.0)
    ), "Result should be equal to 1.0 (i.e. True)"


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
def test_auc_score_2(my_df, method, preprocessor):
    """
    Tests auc_drop when columns are reversed
    """

    kwargs = {}
    if method == "tpot":
        kwargs = {"generations": 5, "population_size": 10}
    my_df2 = my_df.copy()
    score = round(
        auc_score(
            orig_df=my_df[["age", "education", "sex"]],
            mod_df=my_df2[["sex", "education", "age"]],
            sensitive_col="sex",
            method=method,
            modeling_task="binary",
            preprocessor=preprocessor,
            retrain=True,
            **kwargs,
        ),
        3,
    )
    assert score == pytest.approx(1.0), "Result should be equal to 1.0 (i.e. True)"


answer_key = {
    ("logreg", "embeddings"): 0.042,
    ("logreg", "label_encode"): 0,
    ("rfclass", "embeddings"): 0.229,
    ("rfclass", "label_encode"): 0,
    ("tpot", "embeddings"): 0.195,
    ("tpot", "label_encode"): 0.657,
}


@pytest.mark.parametrize("method", ["rfclass"])
@pytest.mark.parametrize("preprocessor", ["label_encode"])
def test_auc_score_3(my_df, method, preprocessor):
    """
    Tests auc_drop when columns are all set to one value
    """
    kwargs = {}
    if method == "tpot":
        kwargs = {"generations": 5, "population_size": 20}
    my_df2 = my_df.copy()
    my_df2["age"] = 1
    my_df2["education"] = "Bachelors"

    score = round(
        auc_score(
            orig_df=my_df[["age", "education", "sex"]],
            mod_df=my_df2[["age", "education", "sex"]],
            sensitive_col="sex",
            method=method,
            preprocessor=preprocessor,
            retrain=True,
            **kwargs,
        ),
        3,
    )
    assert (
        score == answer_key[(method, preprocessor)]
    ), f"{method, preprocessor}: Got {score}, which is not equal to {answer_key[(method, preprocessor)]}."
