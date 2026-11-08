import pytest

from pymasq.datasets import load_census
from pymasq.datasets.data_generator import (
    generate_l_diverse_table,
    generate_t_close_table,
)
from pymasq.errors import InputError
from pymasq.metrics import (
    auc_score,
    beta_likeness,
    indiv_risk_approx,
    indiv_risk_exact,
    is_beta_like,
    is_k_anon,
    is_l_diverse,
    is_t_close,
    k_anon,
    l_diversity,
    t_closeness,
)


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


def test_is_l_diverse():
    df = generate_l_diverse_table(3)
    assert is_l_diverse(df, "sensitive", l_thresh=3)
    assert not is_l_diverse(df, "sensitive", l_thresh=4)


def test_k_anon():
    import pandas as pd
    df = pd.DataFrame({
        "q1": [1, 1, 1, 2, 2],
        "q2": [10, 10, 10, 20, 20],
        "sensitive": ["A", "B", "A", "B", "C"]
    })
    # Group sizes: (1,10) has 3, (2,20) has 2.
    # For k=2, all records meet k-anonymity (size >= 2), 0% violate
    assert k_anon(df, key_vars=["q1", "q2"], k=2) == 0.0
    assert is_k_anon(df, key_vars=["q1", "q2"], k=2)
    # For k=3, group (2,20) has size 2 < 3, so 2 of 5 records violate (40%)
    assert k_anon(df, key_vars=["q1", "q2"], k=3) == pytest.approx(0.4)
    assert not is_k_anon(df, key_vars=["q1", "q2"], k=3)


def test_indiv_risk_formulas():
    # Test indiv_risk_approx for various sample frequencies
    res_eq = indiv_risk_approx(5, 5)
    assert res_eq == pytest.approx(1 / 5)
    res_gt2 = indiv_risk_approx(4, 10)
    assert 0.0 < res_gt2 < 1.0
    res_2 = indiv_risk_approx(2, 10)
    assert 0.0 < res_2 < 1.0
    res_1 = indiv_risk_approx(1, 10)
    assert 0.0 < res_1 < 1.0

    # Test indiv_risk_exact
    res_exact_eq = indiv_risk_exact(5, 5.0)
    assert res_exact_eq == pytest.approx(1 / 5)
    res_exact_2 = indiv_risk_exact(2, 10.0)
    assert 0.0 < res_exact_2 < 1.0
    res_exact_gt2 = indiv_risk_exact(4, 10.0)
    assert 0.0 < res_exact_gt2 < 1.0


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
    kwargs = {"verbose": 0}
    if "tpot" in method:
        kwargs = {"generations": 2, "population_size": 5, "cv": 2, "random_state": 1234}
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
    if "tpot" in method:
        kwargs = {"generations": 2, "population_size": 5, "cv": 2, "random_state": 1234}
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
