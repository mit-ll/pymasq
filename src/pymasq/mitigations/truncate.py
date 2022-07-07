import pandas as pd
import re

from typing import List, Optional, Union, Final

from pymasq import BEARTYPE
from pymasq.config import (
    FORMATTING_ON_OUTPUT,
    FORMATTING_IGNORE_DTYPES,
)
from pymasq.errors import InputError
from pymasq.mitigations.utils import BOTH
from pymasq.utils import formatting

__all__ = [
    "truncate",
    "truncate_by_match",
    "truncate_by_index",
    "INDEX",
    "MATCH",
    "START",
    "END",
]


INDEX: Final = "index"
MATCH: Final = "match"
START: Final = "start"
END: Final = "end"


@formatting(on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=FORMATTING_IGNORE_DTYPES)
@BEARTYPE
def truncate_by_match(
    data: Union[pd.DataFrame, pd.Series],
    match: str,
    keep_before: bool = True,
    ignore_case: bool = False,
    cols: Optional[Union[List, str, int]] = None,
) -> pd.DataFrame:
    """Truncates all characters before or after the first instance of a matching.

    Parameters
    ----------
    data : DataFrame or Series
        The data to be modified.     
    match : str
        The string to search for.
    keep_before : bool, optional (Default: True)
        Truncate all characters before or after `match`.
    ignore_case : bool, optional (Default: False)
        Should the `match` pattern be searched for with exact casing (False) or with casing
        ignored (True)
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.

    Returns
    -------
    DataFrame
        The DataFrame with truncated values.

    Examples
    --------
    >>> df = pymasq.datasets.load_census()
    >>> df = df[['workclass', 'education', 'relationship']].head()
    >>> df
        workclass        education  relationship
    0  State-gov         Bachelors  Not-in-family
    1  Self-emp-not-inc  Bachelors  Husband
    2  Private           HS-grad    Not-in-family
    3  Private           11th       Husband
    4  Private           Bachelors  Wife
    
    >>> truncate_by_match(df[['workclass', 'education', 'relationship']], match='a')
        workclass        education  relationship
    0  St                B          Not-in-f
    1  Self-emp-not-inc  B          Husb
    2  Priv              HS-gr      Not-in-f
    3  Priv              11th       Husb
    4  Priv              B          Wife
    """

    def _truncate_by_match(series, match, ignore_case, keep_before):
        if not ignore_case:
            return series.str.split(re.escape(match), n=1).str[0 if keep_before else -1]
        # same functionality as above, but allows the inclusion of the re.IGNORECASE flag
        return series.apply(
            lambda x: re.split(re.escape(match), x, 1, flags=re.IGNORECASE)
        ).str[0 if keep_before else -1]

    return data.apply(
        _truncate_by_match,
        match=match,
        ignore_case=ignore_case,
        keep_before=keep_before,
    )


@formatting(on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=FORMATTING_IGNORE_DTYPES)
@BEARTYPE
def truncate_by_index(
    data: Union[pd.DataFrame, pd.Series],
    idx: int = 0,
    end: Optional[int] = None,
    trim_from: Optional[str] = None,
    cols: Optional[Union[List, str, int]] = None,
) -> pd.DataFrame:
    """Truncates all characters within an index range or by specific match.

    Truncate all characters within an index range from the start of a string, the back of
    a string, or both sides of a string.

    Parameters
    ----------
    data : DataFrame or Series
        The data to be modified.
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    idx : int (Default: 0)
        The start index to truncate from. If `end` is None, then `idx` will be set to 0.
    end : int, Optional
        The end index to truncate to. If None, `end` will be set to `idx`. If `trim_from` is
        set, `end` will be ignored.
    trim_from : {"start", "end", "both"}, Optional
        The end of the string that should have character trimming applied

    Returns
    -------
    DataFrame
        The DataFrame with truncated values.

    Examples
    --------
    >>> df = pymasq.datasets.load_census()
    >>> df = df[['workclass', 'education', 'relationship']].head()
    >>> df
        workclass        education  relationship
    0  State-gov         Bachelors  Not-in-family
    1  Self-emp-not-inc  Bachelors  Husband
    2  Private           HS-grad    Not-in-family
    3  Private           11th       Husband
    4  Private           Bachelors  Wife
    
    >>> truncate_by_index(df[['workclass', 'education', 'relationship']], idx=1, trim_from='both')
        workclass      education  relationship
    0  tate-go         achelor    ot-in-famil
    1  elf-emp-not-in  achelor    usban
    2  rivat           S-gra      ot-in-famil
    3  rivat           1t         usban
    4  rivat           achelor    if
    """

    def _truncate_by_index(series, trim_from, idx, end):
        if trim_from is None:
            return series.str.slice(idx, end)
        elif trim_from == START:
            return series.str.slice(start=idx)
        elif trim_from == END:
            return series.str.slice(0, -idx)
        elif trim_from == BOTH:
            return series.str.slice(idx, -idx)
        raise InputError(
            f"`trim_from` must be one of ['start', 'end', 'both', None]. (Received: {trim_from})"
        )

    return data.apply(_truncate_by_index, trim_from=trim_from, idx=idx, end=end)


def truncate(
    data: Union[pd.DataFrame, pd.Series], method: str = "index", **kwargs,
) -> pd.DataFrame:
    """Truncate strings by index or after matching a speficic substring.

    Wrapper function for `truncate_by_index` and `truncate_by_match`.

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The data to be modified.
    method : {'index', 'match'}, optional (Default: 'index')
        The truncation method to perform.
    cols : str or list, Optional
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    **kwargs
        Additional arguments to be passed to `truncate_by_index` and `truncate_by_match`.

        If `method` is 'index':

            * idx : int
                The start index to truncate from. If `end` is None, then `idx` will be set to 0.
            * end : int, Optional
                The end index to truncate to. If None, `end` will be set to `idx`.
            * trim_from : {"start", "end", "both"}, Optional
                The end of the string that should have character trimming applied

        If `method` is 'match':

            * match : str
                The string to search for.
            * keep : {"before", "after"}, optional (Default: "before")
                Truncate all characters before or after `match`.
            * ignore_case : boolean, optional (Default: False)
                Should the supplied `match` pattern be case sensitive or not

    Returns
    -------
    DataFrame
        A DataFrame with truncated values.

    Raises
    ------
    InputError
        This error is raised when a `method` is provided either without the required parameters for
        that `method`.
    DataTypeError
        This error is raised when the parameters supplied for a given method do not match the
        appropriate type of the expected parameters of that method.


    See Also
    --------
     pymasq.mitigations.truncate_by_index : Truncates all characters within an index range.

     pymasq.mitigations.truncate_by_match : Truncates all characters before or after the first instance of a matching.


    Examples
    --------
    >>> df = pymasq.datasets.load_census()
    >>> df = df[['workclass', 'education', 'relationship']].head()
    >>> df
        workclass        education  relationship
    0  State-gov         Bachelors  Not-in-family
    1  Self-emp-not-inc  Bachelors  Husband
    2  Private           HS-grad    Not-in-family
    3  Private           11th       Husband
    4  Private           Bachelors  Wife
    
    >>> truncate(df, cols=['workclass', 'education', 'relationship'], method='index', idx=1, trim_from='both')
        workclass      education  relationship
    0  tate-go         achelor    ot-in-famil
    1  elf-emp-not-in  achelor    usban
    2  rivat           S-gra      ot-in-famil
    3  rivat           1t         usban
    4  rivat           achelor    if

    >>> truncate(df[['workclass', 'education', 'relationship']], method='match', match='a')
        workclass        education  relationship
    0  St                B          Not-in-f
    1  Self-emp-not-inc  B          Husb
    2  Priv              HS-gr      Not-in-f
    3  Priv              11th       Husb
    4  Priv              B          Wife
    """
    if method == INDEX:
        return truncate_by_index(
            data,
            idx=kwargs.get("idx", 0),
            end=kwargs.get("end", None),
            trim_from=kwargs.get("trim_from", None),
            cols=kwargs.get("cols", None),
        )
    elif method == MATCH:
        return truncate_by_match(
            data,
            match=kwargs["match"],
            keep_before=kwargs.get("keep_before", True),
            ignore_case=kwargs.get("ignore_case", False),
            cols=kwargs.get("cols", None),
        )

    raise InputError(
        f"Invalid `method` defined; method must be one of ['index', 'match']. (Received: {method}"
    )
