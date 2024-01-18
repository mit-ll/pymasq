import logging

import pandas as pd

from pymasq import config

from pymasq.mitigations import (
    global_recode,
    EQUAL,
    MAGNITUDE,
    LOG_EQUIDISTANT,
)

config.FORMATTING_ON_OUTPUT = True

logger = logging.getLogger(__name__)


def test_global_recode_labels_ordered():
    one_to_ten = range(1, 11)
    series = pd.Series(one_to_ten)

    expected_result = pd.Categorical(
        ["A", "A", "A", "A", "B", "B", "B", "C", "C", "C"],
        dtype="category",
        categories=["A", "B", "C"],
        ordered=True,
    )
    result = global_recode(series, bins=3, ordered=True, labels=["A", "B", "C"])
    logger.info(result)
    assert all(result == expected_result), "This should be true"


def test_global_recode_labels_unordered():
    one_to_ten = range(1, 11)
    series = pd.Series(one_to_ten)

    expected_result = pd.Categorical(
        ["A", "A", "A", "A", "B", "B", "B", "C", "C", "C"],
        dtype="category",
        categories=["A", "B", "C"],
        ordered=False,
    )

    assert all(
        global_recode(series, bins=3, ordered=False, labels=["A", "B", "C"])
        == expected_result
    ), "This should be true"


def test_global_recode_no_labels():
    one_to_ten = range(1, 11)
    series = pd.Series(one_to_ten)
    intervals = [
        pd.Interval(0.99, 2.8, closed="right"),
        pd.Interval(2.8, 4.6, closed="right"),
        pd.Interval(4.6, 6.4, closed="right"),
        pd.Interval(6.4, 8.2, closed="right"),
        pd.Interval(8.2, 10.0, closed="right"),
    ]

    expected_result = pd.Categorical(
        [
            intervals[0],
            intervals[0],
            intervals[1],
            intervals[1],
            intervals[2],
            intervals[2],
            intervals[3],
            intervals[3],
            intervals[4],
            intervals[4],
        ],
        dtype="category",
        ordered=True,
    )
    logger.info(f"EXPECTED RESULT ====>{expected_result}")
    result = global_recode(
        series,
        bins=5,
    )
    logger.info(f"RESULT ====>{result}")

    assert all(result == expected_result), "This should be true"


def test_global_recode_bins_int():
    one_to_ten = range(1, 11)
    series = pd.Series(one_to_ten)
    intervals = [
        pd.Interval(0.99, 2.8, closed="right"),
        pd.Interval(2.8, 4.6, closed="right"),
        pd.Interval(4.6, 6.4, closed="right"),
        pd.Interval(6.4, 8.2, closed="right"),
        pd.Interval(8.2, 10.0, closed="right"),
    ]

    expected_result = pd.Categorical(
        [
            intervals[0],
            intervals[0],
            intervals[1],
            intervals[1],
            intervals[2],
            intervals[2],
            intervals[3],
            intervals[3],
            intervals[4],
            intervals[4],
        ],
        dtype="category",
        ordered=True,
    )

    assert all(global_recode(series, bins=5) == expected_result), "This should be true"


def test_global_recode_bins_list():
    one_to_ten = range(1, 11)
    series = pd.Series(one_to_ten)
    intervals = pd.interval_range(0, 10, 5)

    first = pd.Interval(-0.001, 2.0, closed="right")

    bins = list(range(0, 12, 2))

    expected_result = pd.Categorical(
        [
            first,
            first,
            intervals[1],
            intervals[1],
            intervals[2],
            intervals[2],
            intervals[3],
            intervals[3],
            intervals[4],
            intervals[4],
        ],
        dtype="category",
        ordered=True,
    )

    assert all(
        global_recode(series, bins=bins) == expected_result
    ), "This should be true"


def test_global_recode_bins_intervalindex():
    one_to_ten = range(1, 11)
    series = pd.Series(one_to_ten)
    intervals = pd.interval_range(0, 10, 5)

    expected_result = pd.Categorical(
        [
            intervals[0],
            intervals[0],
            intervals[1],
            intervals[1],
            intervals[2],
            intervals[2],
            intervals[3],
            intervals[3],
            intervals[4],
            intervals[4],
        ],
        dtype="category",
        ordered=True,
    )

    assert all(
        global_recode(series, bins=intervals) == expected_result
    ), "This should be true"


def test_generate_recode_equidistant_no_labels():
    one_to_ten = range(1, 11)
    series = pd.Series(one_to_ten)
    intervals = [
        pd.Interval(0.99, 2.8, closed="right"),
        pd.Interval(2.8, 4.6, closed="right"),
        pd.Interval(4.6, 6.4, closed="right"),
        pd.Interval(6.4, 8.2, closed="right"),
        pd.Interval(8.2, 10.0, closed="right"),
    ]

    expected_result = pd.Categorical(
        [
            intervals[0],
            intervals[0],
            intervals[1],
            intervals[1],
            intervals[2],
            intervals[2],
            intervals[3],
            intervals[3],
            intervals[4],
            intervals[4],
        ],
        dtype="category",
        ordered=True,
    )
    assert all(
        pd.Categorical(global_recode(series, bins=5).values) == expected_result
    ), "This should be True"


def test_generate_recode_log_equidistant_with_unordered_labels():
    one_to_ten = range(1, 11)
    series = pd.Series(one_to_ten)
    expected_result = pd.Categorical(
        ["A", "A", "B", "B", "C", "C", "C", "C", "C", "C"],
        dtype="category",
        categories=["A", "B", "C"],
        ordered=False,
    )

    assert all(
        pd.Categorical(
            global_recode(
                series,
                bin_method=LOG_EQUIDISTANT,
                labels=["A", "B", "C"],
                bins=3,
                ordered=False,
            ).values
        )
        == expected_result
    ), "This should be True"


def test_generate_recode_equal_quantity_no_labels():
    one_to_ten = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    series = pd.Series(one_to_ten)
    intervals = [
        pd.Interval(0.999, 2.8, closed="right"),
        pd.Interval(2.8, 4.6, closed="right"),
        pd.Interval(4.6, 6.4, closed="right"),
        pd.Interval(6.4, 8.2, closed="right"),
        pd.Interval(8.2, 10.0, closed="right"),
    ]

    expected_result = pd.Categorical(
        [
            intervals[0],
            intervals[0],
            intervals[1],
            intervals[1],
            intervals[2],
            intervals[2],
            intervals[3],
            intervals[3],
            intervals[4],
            intervals[4],
        ],
        dtype="category",
        ordered=True,
    )

    assert all(
        pd.Categorical(global_recode(series, bin_method=EQUAL, bins=5).values)
        == expected_result
    ), "This should be True"


def test_generate_recode_order_of_magnitude():
    one_to_hundred = range(1, 101)
    series = pd.Series(one_to_hundred)
    intervals = [
        pd.Interval(0.999, 10.0, closed="right"),
        pd.Interval(10.0, 100.0, closed="right"),
    ]

    expected_result = pd.Categorical(
        [intervals[0]] * 10 + [intervals[1]] * 90, dtype="category", ordered=True
    )

    assert all(
        pd.Categorical(global_recode(series, bin_method=MAGNITUDE, bins=2).values)
        == expected_result
    ), "This should be True"
