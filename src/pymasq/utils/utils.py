import inspect
import functools
import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from typing import Final, List, Optional, Union

from pymasq import BEARTYPE
from pymasq import config
from pymasq.errors import InputError


__all__ = ["BOTH", "as_dataframe", "validate_numeric", "formatting", "freq_calc"]


BOTH: Final = "both"


@BEARTYPE
def as_dataframe(obj, cols: Optional[Union[List, str, int]] = None):
    """ Convert an object data structure into a DataFrame """
    if isinstance(obj, (list, np.ndarray)):
        cols = None
    if cols is not None:
        cols = cols if isinstance(cols, list) else [cols]
    if isinstance(obj, pd.DataFrame):
        if cols is not None:
            return obj.loc[:, cols].copy()
        return obj.copy()
    return pd.DataFrame(obj, columns=cols)


def validate_numeric(
    on_input: bool = config.VALIDATE_NUMERIC_ON_INPUT,
    on_output: bool = config.VALIDATE_NUMERIC_ON_OUTPUT,
):
    def _validate_numeric(func):
        @functools.wraps(func)
        def _validate_numeric_wrapper(data, *args, **kwargs):
            if on_input and not is_numeric_dtype(data.values):
                raise ValueError(
                    f"Detected invalid dtype; {func.__name__} requires numeric dtype. (Received: {data.dtypes})"
                )
            data = func(data, *args, **kwargs)
            if on_output and not is_numeric_dtype(data.values):
                raise ValueError(
                    f"Internal Error: values returned were not numeric. (Received: {data.dtypes})"
                )
            return data

        return _validate_numeric_wrapper

    return _validate_numeric


def formatting(
    on_output: bool = config.FORMATTING_ON_OUTPUT,
    ignore_dtypes: bool = config.FORMATTING_IGNORE_DTYPES,
):
    def _formatting(func):
        @functools.wraps(func)
        def _formatting_wrapper(data, *args, **kwargs):
            func_params = inspect.signature(func).parameters
            input_type = type(data)

            cols = kwargs.get("cols", None)
            if cols is None and "cols" in func_params:
                col_idx = list(func_params).index("cols") - 1
                if col_idx < len(args):
                    cols = args[col_idx]

            data = as_dataframe(data, cols)
            dtypes = data.dtypes
            data = func(data, *args, **kwargs)

            keep_dtypes = kwargs.get("keep_dtypes", None)
            if keep_dtypes is None and "keep_dtypes" in func_params:
                dtypes_idx = list(func_params).index("keep_dtypes") - 1
                if dtypes_idx < len(args):
                    keep_dtypes = args[dtypes_idx]

            if keep_dtypes and not ignore_dtypes:
                try:
                    if not all(data.dtypes == dtypes):
                        data = data.astype(dtypes)
                except:
                    # TODO: switch to logging
                    print("WARNING: Unable to keep original datatypes.")

            if on_output:
                if input_type == pd.Series:
                    return data.squeeze()
                if input_type == np.ndarray:
                    return data.to_numpy()
                if input_type == list:
                    return data.values.tolist()

            return data

        return _formatting_wrapper

    return _formatting


def is_identical(s: pd.Series) -> bool:
    """ Checks if all values in the input series are identical. """
    s = s.to_numpy()  # s.values (pandas<0.24)
    return (s[0] == s).all()


def freq_calc(
    data: pd.DataFrame,
    sensitive_col: Union[int, str],
    quasi_cols: List[Union[int, str]],
    weights: Optional[List[float]] = None,
):
    """
    Parameters
    ----------
    data : pd.DataFrame
        dataFrame of the data for which the frequency counts are being calculated
    cols : string or list of strings, optional (Default: None)
        name of one or more columns for which frequency is being counted
    sensitive_col : str
        The name of the column containing the data that is being obscured by mitigations
    weights = list of floats, optional (Default: None)
        the proportion of rows with the same combination of values in `cols` for
        each row in `data`, only to be used if `data` is a sample.
        If None, weights will be a column of ones indicating that this sample is the population)

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
    freq_df = data.groupby(quasi_cols).count()[sensitive_col]
    freq_df = freq_df.rename("samp_fq").reset_index()

    freqs = pd.merge(data, freq_df, how="outer", on=quasi_cols)
    weights = as_dataframe(weights) if weights else pd.Series([1] * freqs.shape[0])
    freqs["pop_fq"] = freqs["samp_fq"].values * weights

    return freqs
