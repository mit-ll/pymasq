from pymasq.preprocessing.preprocess import LabelEncoder_pm
import pandas as pd
import numpy as np

from typing import Final, List, Optional, Union

from pymasq import BEARTYPE
from pymasq.config import (
    FORMATTING_ON_OUTPUT,
    VALIDATE_NUMERIC_ON_INPUT,
)
from pymasq.errors import InputError
from pymasq.utils import validate_numeric, formatting

__all__ = [
    "global_recode",
    "EQUIDISTANT",
    "LOG_EQUIDISTANT",
    "EQUAL",
    "MAGNITUDE",
    "BIN_FUNCS",
]


EQUIDISTANT: Final = "equidistant"
LOG_EQUIDISTANT: Final = "log_equidistant"
EQUAL: Final = "equal"
MAGNITUDE: Final = "magnitude"


def __gr_equidistant(data: pd.Series, breaks: int) -> pd.Series:
    """ Global Recode for `equidistant` method """
    return np.linspace(data.min(), data.max(), breaks)


def __gr_log_equidistant(data: pd.Series, breaks: int) -> pd.Series:
    """ Global Recode for `log_equidistant` method """
    data_log = np.log(data)
    return np.exp(np.linspace(data_log.min(), data_log.max(), breaks))


def __gr_equal_quantity(data: pd.Series, breaks: int) -> pd.Series:
    """ Global Recode for `equal` method """
    return data.quantile(np.linspace(0, 1, breaks))


def __gr_order_of_magnitude(data: pd.Series, breaks: int) -> pd.Series:
    """ Global Recode for order of `magnitude` method. """
    data_log = np.log10(data)
    return np.power(10, np.linspace(data_log.min(), data_log.max(), breaks))


BIN_FUNCS = {
    EQUIDISTANT: __gr_equidistant,
    LOG_EQUIDISTANT: __gr_log_equidistant,
    EQUAL: __gr_equal_quantity,
    MAGNITUDE: __gr_order_of_magnitude,
}


@formatting(on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=True)
@validate_numeric(on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=False)  # fmt: off
@BEARTYPE
def global_recode(
    data: Union[pd.DataFrame, pd.Series],
    bins: Optional[Union[int, List, np.ndarray, pd.IntervalIndex]] = 2,
    bin_method: Optional[str] = None,
    labels: Optional[List] = None,
    ordered: bool = True,
    ret_ints: bool = False,
) -> pd.DataFrame:
    """Recode numeric values as categorical variables.

    This function modifies a continuous numeric column and recodes it as a categorical variable.

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The data to be modified.
    labels : list of str, optional (Default: None)
        The labels given to the new categories created. Labels must be the
        same length as number of `bins`. If `None` the labels will default to "(a, b]" with a and b
        being the endpoints of the category. Note: if `ordered` is False labels must be provided.
    bins : int, list of floats, or IntervalIndex
        The criteria number of bins to split the data by.
            * int : Defines the number of equal-width bins in the range of `data`. The
              range of `data` is extended by .1% on each side to include the minimum.
              and maximum values of `data`.
            * list of floats : Defines the bin edges allowing for non-uniform
              width. No extension of the range of `data` is done.
            * IntervalIndex : Defines the exact bins to be used. Note that
              IntervalIndex for `bins` must be non-overlapping.
    bin_method : {"equidistant", "log_equidistant", "equal", "magnitude"}, Optional (Default: "equidistant")
        The method for creating breaking up the bins.
        * equidistant : bins have the same width
        * log_equidistant : bins have same width on a natural log scale
        * equal : bins each contain the same amount of items
        * magnitude : bins have the same width on log-10 scale
    ordered : bool, optional (Default: True)
        Whether labels are ordered or not
    ret_ints : bool, optional (Default: False)
        Whether to return data as categoritcal or integer, must be interger to use utility scores

    Returns
    -------
    DataFrame
        A DataFrame with recoded values.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pymasq.mitigations import global_recode
    >>> df = pd.DataFrame(np.random.randint(1, 100, (10, 3)), columns=['a','b','c'])
        a  b   c
    0  27  18  61
    1  25  44  61
    2  82  23  19
    3  62  40  30
    4  37  11  95
    5  46  1   38
    6  20  59  96
    7  46  96  82
    8  72  9   20
    9  12  63  4

    >>> global_recode(df, bins=3)
        a                b                  c
    0  (11.929, 35.333]  (0.904, 32.667]   (34.667, 65.333]
    1  (11.929, 35.333]  (32.667, 64.333]  (34.667, 65.333]
    2  (58.667, 82.0]    (0.904, 32.667]   (3.907, 34.667]
    3  (58.667, 82.0]    (32.667, 64.333]  (3.907, 34.667]
    4  (35.333, 58.667]  (0.904, 32.667]   (65.333, 96.0]
    5  (35.333, 58.667]  (0.904, 32.667]   (34.667, 65.333]
    6  (11.929, 35.333]  (32.667, 64.333]  (65.333, 96.0]
    7  (35.333, 58.667]  (64.333, 96.0]    (65.333, 96.0]
    8  (58.667, 82.0]    (0.904, 32.667]   (3.907, 34.667]
    9  (11.929, 35.333]  (32.667, 64.333]  (3.907, 34.667]

    >>> global_recode(df, bins=3, labels=['lo', 'med', 'hi'])
        a   b    c
    0  lo   lo   med
    1  lo   med  med
    2  hi   lo   lo
    3  hi   med  lo
    4  med  lo   hi
    5  med  lo   med
    6  lo   med  hi
    7  med  hi   hi
    8  hi   lo   lo
    9  lo   med  lo

    >>> global_recode(df, bins=3, bin_method='equal')
        a              b              c
    0  (27.0, 46.0]    (18.0, 44.0]   (30.0, 61.0]
    1  (11.999, 27.0]  (44.0, 96.0]   (30.0, 61.0]
    2  (46.0, 82.0]    (18.0, 44.0]   (3.999, 30.0]
    3  (46.0, 82.0]    (18.0, 44.0]   (30.0, 61.0]
    4  (27.0, 46.0]    (0.999, 18.0]  (61.0, 96.0]
    5  (27.0, 46.0]    (0.999, 18.0]  (30.0, 61.0]
    6  (11.999, 27.0]  (44.0, 96.0]   (61.0, 96.0]
    7  (27.0, 46.0]    (44.0, 96.0]   (61.0, 96.0]
    8  (46.0, 82.0]    (0.999, 18.0]  (3.999, 30.0]
    9  (11.999, 27.0]  (44.0, 96.0]   (3.999, 30.0]

    >>> global_recode(df, bins=3, ret_ints=True)
       a  b  c
    0  0  0  0
    1  0  2  0
    2  2  2  0
    3  2  1  0
    4  0  2  2
    5  1  2  1
    6  2  1  1
    7  2  2  0
    8  1  0  0
    9  0  1  2
    """
    if bin_method is not None:
        if bin_method not in BIN_FUNCS.keys():
            raise InputError(
                f"Invalid `method` defined; method must be one of {BIN_FUNCS.keys()}. (Received: {bin_method})"
            )
        bins = data.apply(BIN_FUNCS[bin_method], breaks=bins + 1)

    data_recode = data.apply(
        lambda col: pd.cut(
            col,
            bins=bins if bin_method is None else bins[col.name],
            labels=labels,
            include_lowest=True,
            ordered=ordered,
            duplicates="drop",
        )
    )
    if ret_ints:
        le = LabelEncoder_pm()
        return le.encode(data_recode.astype(str))

    return data_recode
