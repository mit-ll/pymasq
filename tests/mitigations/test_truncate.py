#!/usr/bin/env python
# coding: utf-8

import logging
import pytest

import pandas as pd

from pymasq.datasets import load_census
from pymasq.mitigations import truncate, INDEX, MATCH, START, END, BOTH

logger = logging.getLogger(__name__)

@pytest.fixture
def my_df():
    df = load_census()
    cols = ["fnlwgt", "education", "marital_status", selected_col, "capital_gain"]
    df = df.loc[:10, cols]
    return df

selected_col: str = "sex"

# ----- Method: Index Tests -----
def test_truncate_index_1(my_df):
    """
    Test that truncate runs correctly for the INDEX method with only a positive idx
    supplied.
    Should only keep characters [0:3)
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], idx=3)
    assert ret[selected_col].isin(["e", "ale"]).all()


def test_truncate_index_2(my_df):
    """
    Test that truncate runs correctly for the INDEX method with only a negative idx
    supplied.
    Should only keep characters [0:-1)
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=INDEX, idx=-1)
    assert ret[selected_col].isin(["e", "e"]).all()


def test_truncate_index_3(my_df):
    """
    Test that truncate runs correctly for the INDEX method with only idx of 0
    supplied and trim_from=END
    Should not keep any characters
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=INDEX, idx=0, trim_from=END)
    assert ret[selected_col].isin([""]).all()


def test_truncate_index_4(my_df):
    """
    Test that truncate runs correctly for the INDEX method with only a very large
    idx supplied. (idx > longest string in the column).
    Should not keep all characters
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=INDEX, end=100)
    assert ret[selected_col].isin(["Male", "Female"]).all()


def test_truncate_index_5(my_df):
    """
    Test that truncate runs correctly for the INDEX method with idx and end
    supplied.
    Should keep characters [1:3)
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=INDEX, idx=1, end=3)
    assert ret[selected_col].isin(["al", "em"]).all()


def test_truncate_index_6(my_df):
    """
    Test that truncate runs correctly for the INDEX method with idx > end
    supplied.
    Should not keep any characters
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=INDEX, idx=3, end=1)
    assert ret[selected_col].isin([""]).all()


def test_truncate_input_7(my_df):
    """
    Test that truncate returns same value if no idx or end supplied
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=INDEX)
    assert ret[selected_col].isin(["Male", "Female"]).all()


# ----- Method: match Tests -----
def test_truncate_match_1(my_df):
    """
    Test that truncate runs correctly for the MATCH method with a pattern that matches a
    part of all strings in the specified column.
    Should only keep characters before "al" for all values
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=MATCH, match="al")
    assert ret[selected_col].isin(["M", "Fem"]).all()


def test_truncate_match_2(my_df):
    """
    Test that truncate runs correctly for the MATCH method with a pattern that matches a
    part of all strings in the specified column.
    Should only keep characters before "em" ("F")for entries with value "Female" and the full entry
    "Male" for the others.
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=MATCH, match="em")
    assert ret[selected_col].isin(["Male", "F"]).all()


def test_truncate_match_3(my_df):
    """
    Test that truncate runs correctly for the MATCH method with a pattern that does not
    match any string in the specified column
    Should keep all characters
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=MATCH, match="cat")
    assert ret[selected_col].isin(["Male", "Female"]).all()


def test_truncate_match_4(my_df):
    """
    Test that truncate runs correctly for the MATCH method with a pattern matches only when
    ignorecase is True
    Should only keep characters before "em" ("F")for entries with value "Female" and the full entry
    "Male" for the others.
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=MATCH, match="EM", ignore_case=True)
    assert ret[selected_col].isin(["Male", "F"]).all()


def test_truncate_match_5(my_df):
    """
    Test pattern matches are properly escaped by the regex expression
    Should only keep characters before "em" ("F")for entries with value "Female" and the full entry
    "Male" for the others.
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=MATCH, match=".*", ignore_case=True)
    assert ret[selected_col].isin(["Male", "Female"]).all()


# ----- Method: More Index Tests -----


def test_truncate_index_11(my_df):
    """
    Test that truncate runs correctly for the INDEX method with only a valid `n` supplied
    Should only keep characters [3:]
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=INDEX, idx=3)
    assert ret[selected_col].isin(["e", "ale"]).all()


def test_truncate_index_12(my_df):
    """
    Test that truncate runs correctly for the INDEX method with a valid `n` supplied
    and trim_from=END
    Should only keep characters [:-3]
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=INDEX, idx=3, trim_from=END)
    assert ret[selected_col].isin(["M", "Fem"]).all()


def test_truncate_index_13(my_df):
    """
    Test that truncate runs correctly for the INDEX method with a valid `n` supplied
    and trim_from=BOTH
    Should only keep characters [1:-1]
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=INDEX, idx=1, trim_from=BOTH)
    assert ret[selected_col].isin(["al", "emal"]).all()


def test_truncate_index_14(my_df):
    """
    Test that truncate runs correctly for the INDEX method with a value of `n` supplied
    greater than some of the string lengths but not others and trim_from=START
    Should only keep the last "e" in "Female"
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=INDEX, idx=5, trim_from=START)
    assert ret[selected_col].isin(["", "e"]).all()


def test_truncate_index_15(my_df):
    """
    Test that truncate runs correctly for the INDEX method with a value of `n` supplied
    greater or equal to than half the length of some strings but not others, and trim_from=BOTH
    Should only keep characters "ma" from "Female"
    """
    ret:pd.DataFrame = truncate(my_df[selected_col], method=INDEX, idx=2, trim_from=BOTH)
    assert ret[selected_col].isin(["", "ma"]).all()


def test_truncate_index_16(my_df):
    """
    Test that truncate runs correctly for the INDEX method with a large value of `n`
    supplied and trim_from=START
    Should not keep any
    """
    ret: pd.DataFrame = truncate(my_df[selected_col], method=INDEX, idx=100, trim_from=START)
    assert ret[selected_col].isin([""]).all()
