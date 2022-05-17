import pandas as pd

from typing import Union, List, Optional, Dict, Final

from pymasq import BEARTYPE
from pymasq.config import (
    FORMATTING_ON_OUTPUT,
    FORMATTING_IGNORE_DTYPES,
    VALIDATE_NUMERIC_ON_INPUT,
    VALIDATE_NUMERIC_ON_OUTPUT,
)
from pymasq.errors import InputError
from pymasq.mitigations.utils import BOTH
from pymasq.utils import formatting, validate_numeric

__all__ = ["top_recoding", "bot_recoding", "topbot_recoding", "TOP", "BOTTOM"]


TOP: Final = "top"
BOTTOM: Final = "bottom"


@formatting(on_output=FORMATTING_ON_OUTPUT)
@validate_numeric(
    on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=VALIDATE_NUMERIC_ON_OUTPUT
)  # fmt: off
@BEARTYPE
def top_recoding(
    data: Union[pd.DataFrame, pd.Series],
    cutoff: Union[int, float],
    to_val: Optional[Union[int, float]] = None,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
) -> pd.DataFrame:
    """Recode all values above a maximum threshold to a specific value.

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The data to be modified.
    cols : listr, str, or int, Optional
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    cutoff : int or float
        Maximum threshold. All values above this threshold will be replaced with `to_val`.
    to_val : int or float, Optional
        Replacement value to use for elements in `data` above `cutoff`.
        If None, then `to_val` will be set to `cutoff`.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Returns
    -------
    DataFrame
        The DataFrame with recoded values.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.random_integers(0, 1000, (10,3)))
        0   1    2
    0  182  529  949
    1  783  375  59
    2  253  349  631
    3  889  683  996
    4  523  554  803
    5  279  379  242
    6  639  26   833
    7  30   307  551
    8  681  885  822
    9  659  180  823

    >>> top_recoding(df, cutoff=800, to_val=999)
        0   1    2
    0  182  529  999
    1  783  375  59
    2  253  349  631
    3  999  683  999
    4  523  554  999
    5  279  379  242
    6  639  26   999
    7  30   307  551
    8  681  999  999
    9  659  180  999
    """
    to_val = cutoff if to_val is None else to_val
    return data.where(data < cutoff, to_val)


@formatting(on_output=FORMATTING_ON_OUTPUT)
@validate_numeric(
    on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=VALIDATE_NUMERIC_ON_OUTPUT
)  # fmt: off
@BEARTYPE
def bot_recoding(
    data: Union[pd.DataFrame, pd.Series],
    cutoff: Union[int, float],
    to_val: Optional[Union[int, float]] = None,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
) -> pd.DataFrame:
    """Recode all values below a minimum threshold to a specific value.

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The data to be modified.
    cutoff : int or float
        Mininum threshold. All values below this threshold will be replaced with `to_val`.
    to_val : int or float, Optional
        Replacement value to use for elements in `data` below `cutoff`.
        If None, then `to_val` will be set to `cutoff`.
    cols : listr, str, or int, Optional
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Returns
    -------
    DataFrame
        The DataFrame with recoded values.

    Example
    -------
    >>> df = pd.DataFrame(np.random.random_integers(0, 1000, (10,3)))
        0   1    2
    0  182  529  949
    1  783  375  59
    2  253  349  631
    3  889  683  996
    4  523  554  803
    5  279  379  242
    6  639  26   833
    7  30   307  551
    8  681  885  822
    9  659  180  823

    >>> bot_recoding(df, cutoff=300, to_val=0)
        0   1    2
    0  0    529  949
    1  783  375  0
    2  0    349  631
    3  889  683  996
    4  523  554  803
    5  0    379  0
    6  639  0    833
    7  0    307  551
    8  681  885  822
    9  659  0    823
    """
    to_val = cutoff if to_val is None else to_val
    return data.where(data > cutoff, to_val)


def topbot_recoding(
    data: Union[pd.Series, pd.DataFrame],
    method: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Recode all values above and/or below a cutoff value to specific values.

    Wrapper function for `top_recoding` and `bot_recoding`.

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The data to be modified.
    method : str {"top","bottom","both"} (Default: "top")
        The recoding method to perform.
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    **kwargs
        Additional arguments to be passed to `top_recoding` and `bot_recoding`.

        if `method` is "top" or "both":
            * top_cutoff : int or float
                Maximum threshold. All values above this threshold will be replaced with `top_to`.
            * top_to : int or float, Optional
                Replacement value to use for elements in `series` above `top_cutoff`.
                If None, then `top_to` will be set to `top_cutoff`.

        if `method` is "bottom" or "both":
            * bot_cutoff : int or float
                Mininum threshold. All values below this threshold will be replaced with `bot_to`.
            * bot_to : int or float, Optional
                Replacement value to use for elements in `series` below `bot_cutoff`.
                If None, then `bot_to` will be set to `bot_cutoff`.


    Returns
    -------
    DataFrame
        A DataFrame with recoded values.

    See Also
    --------
    pymasq.mitigations.top_recoding : Recode all values above a maximum threshold to a specific value.

    pymasq.mitigations.bot_recoding : Recode all values below a minimum threshold to a specific value.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.random_integers(0, 1000, (10,3)))
        0   1    2
    0  182  529  949
    1  783  375  59
    2  253  349  631
    3  889  683  996
    4  523  554  803
    5  279  379  242
    6  639  26   833
    7  30   307  551
    8  681  885  822
    9  659  180  823

    >>> topbot_recoding(df, method="both", top_cutoff=800, top_to=999, bot_cutoff=300, bot_to=0)
        0   1    2
    0  0    529  999
    1  783  375  0
    2  0    349  631
    3  999  683  999
    4  523  554  999
    5  0    379  0
    6  639  0    999
    7  0    307  551
    8  681  999  999
    9  659  0    999
    """
    if method not in [TOP, BOTH, BOTTOM]:
        raise InputError(
            f"Invalid `method` defined; method must be one of ['top', 'bottom', 'both']. (Received: {method})"
        )

    if method in [TOP, BOTH]:
        data = top_recoding(
            data,
            cutoff=kwargs["top_cutoff"],
            to_val=kwargs.get("top_to", None),
            cols=kwargs.get("cols", None),
            keep_dtypes=kwargs.get("keep_dtypes", True),
        )
    if method in [BOTTOM, BOTH]:
        data = bot_recoding(
            data,
            cutoff=kwargs["bot_cutoff"],
            to_val=kwargs.get("bot_to", None),
            cols=kwargs.get("cols", None),
            keep_dtypes=kwargs.get("keep_dtypes", True),
        )
    return data
