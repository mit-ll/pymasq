import pandas as pd
import pytest

from pymasq.errors import InputError
from pymasq.metrics.suda import suda


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "age": [20, 20, 20, 30],
        "zip": [10001, 10001, 10002, 10003],
        "job": ["A", "B", "A", "C"],
    })


def test_suda_basic(sample_data):
    res = suda(sample_data, cols=["age", "zip", "job"])
    assert "suda_score" in res.columns
    assert len(res) == 4
    # Row 3 is unique across multiple single attributes, so highest score
    assert res.loc[3, "suda_score"] == 6.0
    # Row 1 and 2 each have 1 single attribute unique
    assert res.loc[1, "suda_score"] == 2.0
    assert res.loc[2, "suda_score"] == 2.0
    # Row 0 is unique only at 3-attribute combination level
    assert res.loc[0, "suda_score"] == 1.0


def test_suda_empty_df():
    empty_df = pd.DataFrame(columns=["a", "b"])
    res = suda(empty_df, cols=["a", "b"])
    assert "suda_score" in res.columns
    assert len(res) == 0


def test_suda_all_identical():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
    res = suda(df, cols=["a", "b"])
    assert (res["suda_score"] == 0.0).all()


def test_suda_missing_cols(sample_data):
    with pytest.raises(InputError, match="Columns not found"):
        suda(sample_data, cols=["non_existent"])

