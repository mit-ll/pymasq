#!/usr/bin/env python
# coding: utf-8

import pytest

from pymasq import config

config.FORMATTING_ON_OUTPUT = True

from pymasq.datasets import load_census
from pymasq.mitigations import topbot_recoding, TOP, BOTTOM, BOTH
from pymasq.errors import InputError


@pytest.fixture
def my_df():
    df = load_census()
    cols = ["income_level", "age", "education", "race", "sex", "hours_per_week"]
    df = df.loc[:10, cols]
    return df


def test_topbot_1(my_df):
    """
    Test that topbot_recoding runs correctly for a given top_cutoff and bot_cutoff
    and TOP method
    Should replace any values >= 45 with 45
    """
    test_df = topbot_recoding(my_df["age"], top_cutoff=45, bot_cutoff=30, method=TOP)
    assert test_df.max() == 45


def test_topbot_2(my_df):
    """
    Test that topbot_recoding runs correctly for a given top_cutoff and bot_cutoff
    and BOTTOM method
    Should replace any values <= 30 with 30
    """
    test_df = topbot_recoding(my_df["age"], top_cutoff=45, bot_cutoff=30, method=BOTTOM)
    assert test_df.min() == 30


def test_topbot_3(my_df):
    """
    Test that topbot_recoding runs correctly for a given top_cutoff and bot_cutoff
    and BOTH method
    Should replace any values >= 45 and <= 30 with 45 and 30 respectively
    """
    test_df = topbot_recoding(my_df["age"], top_cutoff=45, bot_cutoff=30, method=BOTH)
    assert test_df.max() == 45 and test_df.min() == 30


def test_topbot_4(my_df):
    """
    Test that topbot_recoding runs correctly for a given top_cutoff and bot_cutoff
    and TOP method
    Should replace any values >= 45 with 45 and only that
    """
    test_df = topbot_recoding(my_df["age"], top_cutoff=45, bot_cutoff=30, method=TOP)
    assert not (test_df.max() == 45 and test_df.min() == 30)


def test_topbot_5(my_df):
    """
    Test that topbot throws an InputError if method is not TOP, BOTTOM, or BOTH
    """
    with pytest.raises(InputError):
        topbot_recoding(my_df["age"], top_cutoff=45, bot_cutoff=30, method="test")


def test_topbot_6(my_df):
    """
    Test that topbot_recoding runs correctly for a given top_cutoff and top_to
    and TOP method
    Should replace any values >= 45 with 1000 and only that
    """
    test_df = topbot_recoding(my_df["age"], top_cutoff=30, top_to=1000, method=TOP)
    assert test_df.max() == 1000


def test_topbot_7(my_df):
    """
    Test that topbot_recoding runs correctly for a given bot_cutoff and bot_to
    and BOTTOM method
    Should replace any values <= 30 with -1 and only that
    """
    test_df = topbot_recoding(my_df["age"], bot_cutoff=30, bot_to=-1, method=BOTTOM)
    assert test_df.min() == -1
