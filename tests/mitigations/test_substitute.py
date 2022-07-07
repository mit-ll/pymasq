#!/usr/bin/env python
# coding: utf-8

import pytest

from pymasq.datasets import load_census
from pymasq.mitigations import substitute


@pytest.fixture
def my_df():
    df = load_census()
    cols = ["income_level", "age", "education", "race", "sex", "hours_per_week"]
    df = df.loc[:10, cols]
    return df


def test_substitute_1(my_df):
    """
    Test that substitute runs correctly for a given from_val and to_val
    Should Replace "Male" with "Female"
    """
    test_df = substitute(my_df["sex"], from_val="Male", to_val="Female")
    assert all(test_df.isin(["Female"]))


def test_substitute_2(my_df):
    """
    Test that substitute runs correctly for a given from_val and to_val
    given a substring
    Should Replace "Fem" with "M"
    """
    test_df = substitute(my_df["sex"], from_val="Fem", to_val="M", check_substr=True)
    assert all(test_df.isin(["Male"]))


def test_substitute_3(my_df):
    """
    Test that substitute runs correctly for a given from_val and to_val
    given a substring and ignore case
    Should Replace "Mal" with "X"
    """
    test_df = substitute(
        my_df["sex"], from_val="Mal", to_val="X", check_substr=True, ignore_case=True
    )
    assert all(test_df.isin(["Xe", "FeXe"]))


def test_substitute_4(my_df):
    """
    Test that substitute runs correctly for a given a "*" for from_val
    Should Replace all values in column with "Test"
    """
    test_df = substitute(my_df["sex"], from_val="*", to_val="Test")
    assert all(test_df.isin(["Test"]))


def test_substitute_5(my_df):
    """
    Test that substitute runs correctly for a given a value for from_val
    that does not exist
    Should replace none of the values
    """
    test_df = substitute(my_df["sex"], from_val="Test", to_val="XX")
    assert all(test_df.isin(["Male", "Female"]))


'''
def test_substitute_6(my_df):
    """
    Test that substitute runs correctly for multiple columns
    Should replace "e" with "u"
    """
    test_df = substitute(my_df, ["sex", "race"],
                                from_val="e", to_val="u",
                                check_substr=True)
    assert (test_df.sex.isin(["Malu", "Fumalu"]).all() and
            test_df.race.isin(["Whitu", "Asian-Pac-Islandur", "Black"]).all())
'''


def test_substitute_7(my_df):
    """
    Test that substitute runs correctly for categorical columns
    Should replace "Male" with "Test"
    """
    my_df["sex"] = my_df["sex"].astype("category")
    test_df = substitute(
        my_df["sex"], from_val="Male", to_val="Test", check_substr=True
    )
    assert all(test_df.isin(["Test", "Female"]))
