import pandas as pd

from typing import Union, List, Final, Optional

from pymasq.errors import InputError

__all__ = ["BOTH"]


BOTH: Final = "both"


def _is_identical(s: pd.Series) -> bool:
    """ Checks if all values in the input series are identical. """
    s = s.to_numpy()  # s.values (pandas<0.24)
    return (s[0] == s).all()


def _as_series(
    obj: Union[pd.DataFrame, pd.Series], cols: Optional[Union[str, List[str]]] = None
) -> pd.Series:
    """ Convert an object data structure into a Series """
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
    """ Convert an object data structure into a DataFrame """
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
    freq_df = df.groupby(cols).count()[sensitive_col]
    freq_df = freq_df.rename("samp_fq")
    freq_df = freq_df.reset_index()
    result = pd.merge(df, freq_df, how="outer", on=cols)
    result["pop_fq"] = result["samp_fq"].values * weights

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
