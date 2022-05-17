import math
import pandas as pd

from typing import List, Union, Optional

from pymasq import BEARTYPE
from pymasq.config import (
    FORMATTING_ON_OUTPUT,
    FORMATTING_IGNORE_DTYPES,
    VALIDATE_NUMERIC_ON_INPUT,
    VALIDATE_NUMERIC_ON_OUTPUT,
)
from pymasq.utils import formatting, validate_numeric

__all__ = ["rounding"]


@formatting(
    on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=FORMATTING_IGNORE_DTYPES
)  # fmt: off
@validate_numeric(
    on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=VALIDATE_NUMERIC_ON_OUTPUT
)  # fmt: off
@BEARTYPE
def rounding(
    data: Union[pd.DataFrame, pd.Series],
    magnitude: int = 0,
    round_decimal: bool = False,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
) -> pd.DataFrame:
    """ Round numerical values to the nearest place value.

    Round to the nearest whole number or decimal. Values are always rounded up.

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The data to be modified.    
    magnitude : int (Default: 0)
        The place value to round to.
    round_decimal : bool (Default: False)
        If `True`, round to the nearest decimal place value. Else, round to nearest whole number.
    cols : listr, str, or int, Optional
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Returns
    -------
    DataFrame
        A DataFrame with rounded values.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.uniform(0.0, 1000, (10,3)))    
       0           1           2
    0  790.885012  378.955986  598.524492
    1  396.506198  416.688230  801.133469
    2  68.949521   487.262995  767.432916
    3  708.297028  414.632973  957.846657
    4  815.998951  539.653967  144.731417
    5  533.075232  281.206489  126.433357
    6  38.475080   41.117392   187.707412
    7  563.563829  545.063224  170.020492
    8  193.475847  216.862268  767.035219
    9  327.110469  903.003675  240.985534

    >>> rounding(df, magnitude=-2, cols=[1,2])
       1       2
    0  378.96  598.52
    1  416.69  801.13
    2  487.26  767.43
    3  414.63  957.85
    4  539.65  144.73
    5  281.21  126.43
    6  41.12   187.71
    7  545.06  170.02
    8  216.86  767.04
    9  903.00  240.99
    """
    magnitude = magnitude if round_decimal else -1 * magnitude
    return data.round(magnitude)
