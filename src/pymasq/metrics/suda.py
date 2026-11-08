import math
from itertools import combinations
from typing import Any, Dict, List, Optional

import pandas as pd

from pymasq import BEARTYPE
from pymasq.errors import InputError

__all__ = ["suda"]


@BEARTYPE
def suda(
    df: pd.DataFrame,
    cols: List[str],
    max_msu: Optional[int] = None,
    **kwargs: Dict[Any, Any],
) -> pd.DataFrame:
    """Special Uniques Detection Algorithm (SUDA) for disclosure risk assessment.

    Identifies Minimal Sample Uniques (MSUs) across subsets of quasi-identifier
    columns up to size `max_msu` and calculates an individual SUDA risk score
    for each record based on factorial weighting of its minimal unique subsets.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to evaluate.
    cols : List[str]
        Quasi-identifier columns to inspect for sample uniqueness.
    max_msu : int, optional
        Maximum size of attribute subsets to consider (Default: min(len(cols), 3)).

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with a 'suda_score' column added.

    Raises
    ------
    InputError
        If any column in `cols` is not present in `df`.
    """
    if not set(cols).issubset(df.columns):
        missing = [c for c in cols if c not in df.columns]
        raise InputError(f"Columns not found in DataFrame: {missing}")

    K = len(cols)
    if K == 0 or len(df) == 0:
        result = df.copy()
        result["suda_score"] = 0.0
        return result

    max_k = min(K, max_msu) if max_msu is not None else min(K, 3)
    scores = pd.Series(0.0, index=df.index)

    # Track MSUs per row: mapping row index -> set of frozensets of column names that are unique
    row_uniques = {idx: set() for idx in df.index}

    for m in range(1, max_k + 1):
        weight = float(math.factorial(max_k - m))
        for subset in combinations(cols, m):
            counts = df.groupby(list(subset), observed=False)[subset[0]].transform("size")
            unique_indices = counts[counts == 1].index

            for idx in unique_indices:
                sub_frozenset = frozenset(subset)
                if not any(existing.issubset(sub_frozenset) for existing in row_uniques[idx]):
                    row_uniques[idx].add(sub_frozenset)
                    scores[idx] += weight

    result = df.copy()
    result["suda_score"] = scores
    return result
