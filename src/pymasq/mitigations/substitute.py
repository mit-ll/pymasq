import pandas as pd
import re

from typing import List, Optional, Union

from pymasq import BEARTYPE
from pymasq.config import FORMATTING_ON_OUTPUT, FORMATTING_IGNORE_DTYPES
from pymasq.utils import formatting

__all__ = ["substitute"]


def __format_if_list(
    from_val: Union[str, float, int, List], to_val: Union[str, float, int, List]
):
    """ Format input values if at least one of them is a list """
    if isinstance(from_val, list):
        to_val = to_val if isinstance(to_val, list) else [to_val]
        if len(to_val) == 1:
            to_val = [to_val[0] for _ in range(len(from_val))]
        # TODO: elif not of same length, raise
    elif isinstance(to_val, list):
        from_val = from_val if isinstance(from_val, list) else [from_val]
        if len(from_val) == 1:
            from_val = [from_val[0] for _ in range(len(to_val))]
        # TODO: elif not of same length, raise

    return from_val, to_val


@formatting(on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=True)
@BEARTYPE
def substitute(
    data: Union[pd.DataFrame, pd.Series],
    from_val: Union[str, float, int, List],
    to_val: Union[str, float, int, List],
    check_substr: bool = False,
    replace_all: bool = False,
    ignore_case: bool = False,
    cols: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Substitute, or replace, a specific value with another.

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The data to be modified.
    from_val : str, numeric, or list
        The value to search for and replace.

        * str, numeric:
            If `from_val` is a string or numeric value, then `to_val` must be a single string or numeric value, or a list of unit length.
            If "*" passed, then all values will be replaced.
            If a regular expression string is passed, refer to [1]_.
        * list:
            If `from_val` is a list, then `to_val` must be a single string or numeric value, or a list of unit or equal length.
    to_val : str, numeric, or list
        The value to replace `from_val` with.

        * str, numeric:
            If `to_val` is a string or numeric value, then `from_val` must be a single string or numeric value, or a list of unit length.
            If a regular expression string is passed, refer to [1]_.
        * list:
            if `to_val` is a list, then `from_val` must be a single string or numeric value, or a list of unit or equal length.
    check_substr : bool, optional (Default: False)
        Check if a string value contains `from_val` as a substring.
        This will be ignored if `from_val` and/or `to_val` are lists.
    replace_all: bool, optional (Default: False)
        Replace an entire string value with `to_val`. Note that this will always be
        True for exact matches (e.g., `check_substr` is False).
    ignore_case : bool, optional (Default: False)
        Ignore the case of the string being substituted.
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    **kwargs : dict
        Additional arguments to pass to [1]_.

    Returns
    -------
    DataFrame
        A DataFrame with substituted values.

    Note
    ----
    Regular expressions can be passed as `from_val` and `to_val`, in which case the functionality
    of `pymasq.mitigations.substitute` will be defered in favor of `pandas.Series.str.replace` [1]_.

    References
    ----------
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.replace.html

    Examples
    --------
    >>> df = pymasq.datasets.load_census()
    >>> df = df[['education', 'race', 'sex']].head(10)
    >>> df
        workclass        sex
    0  State-gov         Male
    1  Self-emp-not-inc  Male
    2  Private           Male
    3  Private           Male
    4  Private           Female
    5  Private           Female
    6  Private           Female
    7  Self-emp-not-inc  Male
    8  Private           Female
    9  Private           Male

    >>> substitute(df, from_val="em", to_val="EM") # no match
        workclass        sex
    0  State-gov         Male
    1  Self-emp-not-inc  Male
    2  Private           Male
    3  Private           Male
    4  Private           Female
    5  Private           Female
    6  Private           Female
    7  Self-emp-not-inc  Male
    8  Private           Female
    9  Private           Male

    >>> substitute(df, from_val="em", to_val="M", check_substr=True)
        workclass        sex
    0  State-gov         Male
    1  Self-EMp-not-inc  Male
    2  Private           Male
    3  Private           Male
    4  Private           FEMale
    5  Private           FEMale
    6  Private           FEMale
    7  Self-EMp-not-inc  Male
    8  Private           FEMale
    9  Private           Male

    >>> substitute(df, from_val="em", to_val="EM", check_substr=True, replace_all=True)
        workclass  sex
    0  State-gov   Male
    1  EM          Male
    2  Private     Male
    3  Private     Male
    4  Private     EM
    5  Private     EM
    6  Private     EM
    7  EM          Male
    8  Private     EM
    9  Private     Male

    >>> substitute(df, from_val="ma", to_val="MA", check_substr=True, replace_all=True, ignore_case=True)
        workclass        sex
    0  State-gov         MA
    1  Self-emp-not-inc  MA
    2  Private           MA
    3  Private           MA
    4  Private           MA
    5  Private           MA
    6  Private           MA
    7  Self-emp-not-inc  MA
    8  Private           MA
    9  Private           MA
    """
    from_val, to_val = __format_if_list(from_val, to_val)

    def _substitute(series, check_substr, replace_all, ignore_case, **kwargs):
        if isinstance(from_val, list) and isinstance(to_val, list):
            # recursively apply `substitute` to each (from_val, to_val) pair
            data = series
            for _from_val, _to_val in zip(from_val, to_val):
                data = substitute(
                    data,
                    from_val=_from_val,
                    to_val=_to_val,
                    check_substr=check_substr,
                    replace_all=replace_all,
                    ignore_case=ignore_case,
                    **kwargs,
                )
            if isinstance(data, pd.DataFrame):
                return data.squeeze()

            return data

        if from_val == "*":
            # replace all values
            return series.replace(series.values, to_val)

        if pd.api.types.is_numeric_dtype(series):
            return series.replace(from_val, to_val)

        ignore_case = not ignore_case

        if check_substr:
            if replace_all:
                # replace entire value if substr is found
                return series.where(
                    ~series.str.contains(from_val, case=ignore_case, **kwargs), to_val
                )
            # replace only the substr in the value
            return series.str.replace(from_val, to_val, case=ignore_case, **kwargs)
        # replace entire value if exact match is found
        return series.str.replace(
            r"\b%s\b" % (from_val), to_val, case=ignore_case, regex=True, **kwargs
        )

    return data.apply(
        _substitute,
        check_substr=check_substr,
        replace_all=replace_all,
        ignore_case=ignore_case,
        **kwargs,
    )
