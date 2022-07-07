from typing import Union, List

import pandas as pd
import numpy as np

from .utils import _as_series

__all__ = ["rank_swap"]


def rank_swap(
    data: Union[pd.DataFrame, pd.Series],
    cols: Union[str, List[str]] = None,
    **kwargs
) -> pd.Series:
    """ TODO

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The data to be modified.
    cols : str or list, Optional
        The name of the column or columns to subset from `data` if `data` is a dataframe.


    """
    series = _as_series(data, cols)

    return series
