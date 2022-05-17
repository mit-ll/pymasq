from pymasq.config import CATEGORY_THRESHOLD
import pandas as pd

from random import getrandbits
from pandas.api.types import is_numeric_dtype
from pymasq import BEARTYPE

__all__ = ["uniq_col_name", "_get_model_task"]


@BEARTYPE
def uniq_col_name(df, prefix: str = "class") -> str:
    """
    Helper function to return a column name that is unique given
    existing column names in the dataframe.

    Uniqueness is provided via a 32-bit random int as a suffix

    Parameters
    ----------
    df: pd.dataFrame
        A dataframe with a set of columns
    prefix: str, optional (Default: "class")
        the prefix of the new column name.

    Return
    ------
    str:
        The new column name as a string. The df is not modified
    """
    class_col = f"{prefix}{getrandbits(32)}"
    while class_col in df.columns:
        class_col = f"{prefix}{getrandbits(32)}"
    return class_col


@BEARTYPE
def _get_model_task(
    sensitive_col: pd.Series, cat_threshold: int = CATEGORY_THRESHOLD
) -> str:
    """
    Sets modeling type based on how many unique values are present and data types.

    Parameters
    ----------
    df: pd.dataFrame
        A dataframe with a set of columns
    prefix: str, optional (Default: "class")
        the prefix of the new column name.

    Return
    ------
    str:
        The modeling type as a string. The df is not modified
    """
    num_unique = sensitive_col.nunique()
    if num_unique == 2:
        return "binary"
    elif num_unique < cat_threshold:
        return "multi_class"
    elif is_numeric_dtype(sensitive_col):
        return "regression"
    else:
        print(
            "The number of unique categories: {} is greater than the threshold of {} and is dtype {}".format(
                num_unique, cat_threshold, sensitive_col.dtype
            )
        )
        return None
