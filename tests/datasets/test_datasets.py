import pytest

from pymasq.datasets import load_census, load_data, load_loan
from pymasq.errors import InputError


def test_load_census():
    df = load_census()
    assert len(df) > 0
    assert "sex" in df.columns


def test_load_loan():
    df = load_loan()
    assert len(df) > 0


def test_load_data_valid():
    df = load_data("prestige.csv")
    assert len(df) > 0


def test_load_data_path_traversal():
    with pytest.raises(InputError, match="Path traversal detected"):
        load_data("../../etc/passwd")

