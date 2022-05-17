import numpy as np
import pandas as pd
import pytest
from pymasq.datasets import load_census
from pymasq.errors import DataTypeError, InputError
from pymasq.sa import *
from sklearn.linear_model import LinearRegression


@pytest.fixture
def numerical_df():
    new_df = pd.DataFrame(
        {
            "age": np.random.randint(low=18, high=65, size=100),
            "hair_color": np.random.randint(5, size=100),
            "region": np.random.randint(3, size=100),
            "weekly_hrs": np.random.randint(low=10, high=60, size=100),
        }
    )
    return new_df


@pytest.fixture
def sample_model(numerical_df):
    model = LinearRegression()
    salary = (
        numerical_df["weekly_hrs"] / 40 * 40000
        + numerical_df["region"] * 10000
        + numerical_df["age"] / 10 * (1 + numerical_df["region"]) * 1000
        + np.round(5000 * np.random.random(size=100), 2)
    )
    model.fit(numerical_df, salary)
    return model


def test_gen_params_1():
    """
    Should error because the input dataframe is not all numeric
    """
    with pytest.raises(DataTypeError):
        census = load_census()
        gen_params(census)


def test_gen_params_2(numerical_df):
    """
    Tests that the returned dictionary has the correct keys
    """
    params = gen_params(numerical_df)
    assert list(params.keys()) == ["num_vars", "names", "bounds"]


def test_gen_samples_1(numerical_df):
    """
    Should error because the input method is not "uniform" or "sample"
    """
    with pytest.raises(InputError):
        gen_samples(numerical_df, method="bad_method")


def test_gen_samples_2(numerical_df):
    """
    Verifies the number of samples generated is what we expect for the uniform method - with
    second order calculations (𝑁*(2𝐷+2))
    """
    num_samples = 1000
    samples = gen_samples(
        numerical_df, method="uniform", num_samples=num_samples, calc_second_order=True
    )
    assert len(samples) == num_samples * (2 * len(numerical_df.columns) + 2)


def test_gen_samples_3(numerical_df):
    """
    Verifies the number of samples generated is what we expect for the uniform method - without
    second order calculations (𝑁*(𝐷+2))
    """
    num_samples = 1000
    samples = gen_samples(
        numerical_df, method="uniform", num_samples=num_samples, calc_second_order=False
    )
    assert len(samples) == num_samples * (len(numerical_df.columns) + 2)


def test_analyze_1(numerical_df, sample_model):
    """
    Tests that a default run of analyze will work
    """
    params = gen_params(numerical_df)
    samples = gen_samples(numerical_df, params=params)
    s1, st, s2 = analyze(sample_model, params, samples)
    assert (
        s1.size == params["num_vars"]
        and st.size == params["num_vars"]
        and s2.shape == (params["num_vars"], params["num_vars"])
    )


def test_analyze_2(numerical_df, sample_model):
    """
    Tests that analyze will work without including second order computations
    """
    params = gen_params(numerical_df)
    samples = gen_samples(numerical_df, params=params, calc_second_order=False)
    s1, st, s2 = analyze(sample_model, params, samples, calc_second_order=False)
    assert (
        s1.size == params["num_vars"] and st.size == params["num_vars"] and s2 is None
    )
