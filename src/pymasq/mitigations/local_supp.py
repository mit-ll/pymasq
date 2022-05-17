import pandas as pd

from typing import Any, List, Optional, Union

from pymasq import BEARTYPE
from pymasq.config import (
    FORMATTING_ON_OUTPUT,
)
from pymasq.errors import InputError, NotInRangeError
from pymasq.metrics.risk_scores import indiv_risk
from pymasq.utils import formatting


__all__ = ["local_supp"]


@formatting(on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=True)  # fmt: off
@BEARTYPE
def local_supp(
    data: Union[pd.DataFrame, pd.Series],
    suppress_col: Union[str, int],
    sensitive_col: Union[str, int],
    to_val: Any,
    method: str = "approx",
    risk_threshold: float = 0.15,
    max_unique: int = 20,
    weights: Optional[List[Union[float, int]]] = None,
    qual: Union[int, float] = 1,
    keep_dtypes: bool = True,
):
    """Suppress values to achieve k-anonymity.

    This function suppresses values in the given column where the record or row
    poses a risk over the threshold.

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The data to be modified.
    suppress_col : str or int
        Column that will be modified to obfuscate it relationship with `sensitive_col`.
    sensitive_col : str or int
        Column whose relationship to other columns needs to be obfuscated.
    to_val : Any
        Value with which to replace suppressed values.
    method : {'approx', 'exact'} (Default: 'approx')
        Precision for calculating the individual risk of each value.
    risk_threshold : float [0, 1], optional (Default: 0.15)
        Threshold for to determine which values of `suppress_col` will be suppressed.
    max_unique : int, optional (Default: 20)
        Maximum number of unique values allowed per column. Large number of unique values
        may become computationally expensive for calculating individual risk.
    weights : list, optional (Default: None)
        Proportion of rows with the same combination of values in `data` for
        each row in `data`, only to be used if `data` is a sample/subset of a larger dataset.
        If None, `weights` will be a unit vector indicating that `data` is the full population.
    qual : float (0, 1], optional (Default: 1)
        Perceived quality of frequency counts to act as a final correction factor.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Returns
    -------
    DataFrame
        A DataFrame with suppressed values.

    Raises
    ------
    InputError:
        Raised when invalid arguments are passed.
    NotInRangeError
        Raised if `risk_threshold` is outside the interval [0, 1].

    Examples
    --------
    >>> df = pymasq.datasets.load_census()
    >>> df = df[['workclass', 'education', 'relationship', 'race', 'sex']].head(10)
    >>> df
       workclass         education  race   sex
    0  State-gov         Bachelors  White  Male
    1  Self-emp-not-inc  Bachelors  White  Male
    2  Private           HS-grad    White  Male
    3  Private           11th       Black  Male
    4  Private           Bachelors  Black  Female
    5  Private           Masters    White  Female
    6  Private           9th        Black  Female
    7  Self-emp-not-inc  HS-grad    White  Male
    8  Private           Masters    White  Female
    9  Private           Bachelors  White  Male

    >>> local_supp(df, suppress_col='education', sensitive_col='race', to_val='NA', risk_threshold=0.6, cols=['workclass', 'education', 'race'])
       workclass         education  race
    0  State-gov         NA         White
    1  Self-emp-not-inc  NA         White
    2  Private           NA         White
    3  Private           NA         Black
    4  Private           Bachelors  Black
    5  Private           Masters    White
    6  Private           NA         Black
    7  Self-emp-not-inc  NA         White
    8  Private           Masters    White
    9  Private           Bachelors  White

    >>> local_supp(df, suppress_col="education", sensitive_col="race", to_val='NA', method='exact', risk_threshold=0.6)
       workclass         education  race   sex
    0  State-gov         NA         White  Male
    1  Self-emp-not-inc  NA         White  Male
    2  Private           NA         White  Male
    3  Private           NA         Black  Male
    4  Private           NA         Black  Female
    5  Private           Masters    White  Female
    6  Private           NA         Black  Female
    7  Self-emp-not-inc  NA         White  Male
    8  Private           Masters    White  Female
    9  Private           NA         White  Male
    """
    if not (0 <= risk_threshold <= 1):
        raise NotInRangeError(
            f"Risk threshold (`threshold`) must be between 0 and 1, inclusive. (Received: {risk_threshold})"
        )

    if not all(col in data.columns for col in [sensitive_col, suppress_col]):
        raise InputError(
            f"Values of `sensitive_col` and/or `suppress_col` are not in `data` columns {list(data.columns)}. (Received: {sensitive_col} and {suppress_col})"
        )

    quasi_identifiers = [
        x for x in data.columns if x != sensitive_col and data[x].nunique() < max_unique
    ]

    risks = indiv_risk(
        data,
        sensitive_col,
        quasi_identifiers,
        weights=weights,
        method=method,
        qual=qual,
    )
    if not keep_dtypes and type(to_val) != type(data.loc[0, suppress_col]):
        # TODO: switch to logging
        print(
            f"WARNING: The datatype of the `suppress_col` ({suppress_col}`) will be changed."
        )

    # Adds new category to Categorical columns in data
    for col in data.columns:
        if isinstance(data[col].dtype, pd.CategoricalDtype):
            data[col] = data[col].cat.add_categories(to_val)

    data.loc[risks > risk_threshold, suppress_col] = to_val

    return data
