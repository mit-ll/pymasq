from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from pymasq import BEARTYPE
from pymasq.config import (
    DEFAULT_SEED,
    FORMATTING_IGNORE_DTYPES,
    FORMATTING_ON_OUTPUT,
    VALIDATE_NUMERIC_ON_INPUT,
    VALIDATE_NUMERIC_ON_OUTPUT,
)
from pymasq.errors import InputError, NotInRangeError
from pymasq.utils import formatting, validate_numeric

__all__ = ["rank_swap"]


@formatting(
    on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=FORMATTING_IGNORE_DTYPES
)
@validate_numeric(
    on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=VALIDATE_NUMERIC_ON_OUTPUT
)
@BEARTYPE
def rank_swap(
    data: Union[pd.DataFrame, pd.Series],
    p: Union[int, float] = 0.1,
    cols: Optional[Union[List, str, int]] = None,
    seed: Optional[int] = None,
    keep_dtypes: bool = True,
) -> pd.DataFrame:
    """Perform rank swapping on numerical columns.

    Rank swapping is a perturbative statistical disclosure control (SDC) technique
    for continuous or ordinal data. Records are sorted by their values to establish
    ranks. For each record with rank i, a partner with rank j is randomly chosen
    within a bounded interval [i - P, i + P], where P is defined as a proportion
    or percentage of the total number of records (P = max(1, floor(p * N))), and
    their values are swapped.

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The data to be modified.
    p : int or float (Default: 0.1)
        The rank swap range parameter. If 0 < p <= 1, it represents the fraction of
        total records. If 1 < p <= 100, it represents the percentage of total records.
        Must be in (0, 100].
    cols : list, str, or int, Optional
        The name of the column or columns to modify if `data` is a DataFrame.
        If None, all columns in `data` are swapped.
    seed : int, Optional
        Seed for the random number generator. If None, `DEFAULT_SEED` is used.
    keep_dtypes : bool (Default: True)
        Determine whether output column dtypes should match input column dtypes.

    Returns
    -------
    DataFrame
        A DataFrame with rank-swapped values.

    Raises
    ------
    NotInRangeError
        If `p` is <= 0 or > 100.
    """
    if p <= 0 or p > 100:
        raise NotInRangeError(
            f"Swap range parameter `p` must be in (0, 100]. (Received: {p})"
        )

    # Normalize percentage (1 < p <= 100) to fraction
    p_frac = p / 100.0 if p > 1.0 else float(p)

    rng = np.random.default_rng(DEFAULT_SEED if seed is None else seed)
    df = data.copy()

    for col in df.columns:
        if not is_numeric_dtype(df[col].dtype):
            raise InputError(f"Column '{col}' must be numeric to perform rank swapping.")

        vals = df[col].to_numpy().copy()
        valid_mask = ~pd.isna(vals)
        valid_idx = np.flatnonzero(valid_mask)
        N = len(valid_idx)
        if N <= 1:
            continue

        P = max(1, int(np.floor(N * p_frac)))
        sub_vals = vals[valid_idx]
        order = np.argsort(sub_vals)

        for r in range(N):
            low = max(0, r - P)
            high = min(N, r + P + 1)
            r2 = rng.integers(low, high)
            if r != r2:
                idx1 = order[r]
                idx2 = order[r2]
                sub_vals[idx1], sub_vals[idx2] = sub_vals[idx2], sub_vals[idx1]

        vals[valid_idx] = sub_vals
        df[col] = vals

    return df
