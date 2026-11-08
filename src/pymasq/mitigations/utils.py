from typing import Final, List, Optional, Union

import numpy as np
import pandas as pd

from pymasq.errors import InputError

__all__ = ["BOTH"]


BOTH: Final = "both"


def _is_identical(s: pd.Series) -> bool:
    """Checks if all values in the input series are identical."""
    if len(s) == 0:
        return True
    arr = s.to_numpy()
    if pd.isna(arr[0]):
        return bool(pd.isna(arr).all())
    return bool((arr[0] == arr).all())


def _as_series(
    obj: Union[pd.DataFrame, pd.Series], cols: Optional[Union[str, List[str]]] = None
) -> pd.Series:
    """Convert an object data structure into a Series"""
    if isinstance(obj, pd.DataFrame):
        if cols is None:
            raise InputError(
                f"Failed to convert input DataFrame to a Series; `cols` was not defined. (Received: {cols})"
            )
        if isinstance(cols, list):
            if len(cols) > 1:
                raise InputError(
                    f"Failed to convert input DataFrame to a Series; `cols` must be of unit length. (Received: {cols})"
                )
            cols = cols[0]
        return obj[cols].copy()
    return pd.Series(obj)


def _as_dataframe(
    obj: Union[pd.DataFrame, pd.Series], cols: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """Convert an object data structure into a DataFrame"""
    if isinstance(obj, pd.DataFrame):
        if cols is None:
            return obj.copy()
        cols = cols if isinstance(cols, list) else [cols]
        return obj[cols].copy()
    return pd.DataFrame(obj, columns=cols)


def __calc_freq(
    df: pd.DataFrame,
    cols: Union[str, List[str]],
    sensitive_col: str,
    weights: List[float],
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame
        dataFrame of the data for which the frequency counts are being calculated
    cols : string or list of strings
        name of one or more columns for which frequency is being counted
    sensitive_col : str
        The name of the column containing the data that is being obscured by mitigations
    weights = list of floats, optional
        the proportion of rows with the same combination of values in `cols` for
        each row in `df`, only to be used if `df` is a sample
        (Default: None, if None weights will be a column of ones indicating that
         this sample is the population)

    Returns
    -------
    DataFrame
        dataframe with columns 'samp_fq' for sample frequency and 'pop_fq' for population frequency

    Examples
    --------
    TODO

    """
    cols_list = cols if isinstance(cols, list) else [cols]
    result = df.copy()
    result["samp_fq"] = df.groupby(cols_list, observed=False)[cols_list[0]].transform("size")
    if isinstance(weights, (pd.Series, pd.DataFrame)):
        w = weights.to_numpy().flatten()
    else:
        w = np.asarray(weights).flatten()
    result["pop_fq"] = result["samp_fq"].to_numpy() * w

    return result


def freq_calc(
    df: pd.DataFrame,
    cols: Optional[Union[str, List[str]]] = None,
    sensitive_col: Optional[str] = None,
    weights: Optional[List[float]] = None,
):
    """
    Parameters
    ----------
    df : pd.DataFrame
        dataFrame of the data for which the frequency counts are being calculated
    cols : string or list of strings, optional
        name of one or more columns for which frequency is being counted
        (Default: None)
    sensitive_col : str
        The name of the column containing the data that is being obscured by mitigations
    weights = list of floats, optional
        the proportion of rows with the same combination of values in `cols` for
        each row in `df`, only to be used if `df` is a sample
        (Default: None, if None weights will be a column of ones indicating that
         this sample is the population)

    Returns
    -------
    DataFrame
        dataframe with columns 'samp_fq' for sample frequency and 'pop_fq' for population frequency

    Raises
    ------
    InputError:
        This error is raised when invalid arguments are passed.

    Examples
    --------
    TODO

    """
    df_cols = cols + [sensitive_col] + ["order"]
    df = _as_dataframe(df, df_cols)
    weights = _as_series(weights) if weights else _as_series([1] * df.shape[0])
    counts_df = __calc_freq(df, cols, sensitive_col, weights)

    return counts_df
