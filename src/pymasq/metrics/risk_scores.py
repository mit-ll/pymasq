from typing import List, Callable, Dict, Union, Final
from pymasq.errors import InputError, NotInRangeError

import numpy as np
import pandas as pd

from copy import copy
from typing import Callable, Dict, Final, List, Optional, Union

from pymasq import BEARTYPE
from pymasq.errors import InputError, NotInRangeError
from pymasq.utils import formatting, freq_calc


__all__ = [
    "k_anon",
    "is_k_anon",
    "is_k_anon_col",
    "l_diversity",
    "is_l_diverse",
    "t_closeness",
    "is_t_close",
    "beta_likeness",
    "is_beta_like",
    "indiv_risk",
    "indiv_risk_approx",
    "indiv_risk_exact",
    "DISTINCT",
    "ENTROPY",
    "NUMERIC",
    "CATEGORICAL",
    "APPROX",
    "EXACT",
]


DISTINCT: Final = "distinct"
ENTROPY: Final = "entropy"
NUMERIC: Final = "numeric"
CATEGORICAL: Final = "categorical"
APPROX: Final = "approx"
EXACT: Final = "exact"


@BEARTYPE
def k_anon(
    df: pd.DataFrame,
    key_vars: Optional[List[str]] = None,
    k: int = 5,
    sensitive_col: str = "",
    label: str = "",
) -> float:
    """
    This function calculates the percent of records, or rows, in a data frame that violate
    k-anonymity, such that a record that violates k-anonymity contains duplicate values within
    the key_vars columns such that when the group is counted it is less than or equal to k as
    defined by the user.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame containing records used to calculate k-anonymity
    key_vars : List[str], optional
        A list of the column names from the data frame (df) that will be included in the k-anonymity calculation
        (Default: None)
    k : int, optional
        An integer for k that denotes the threshold for how many unique samples can be in a group
        (Default: 5)
    sensitive_col : str, optional
        A string of the column name in the data frame that contains the labels for groups, if not included then the
        k-anonymity will be calculated for the whole data frame
        (Default: "")
    label : str, optional
        A string for the group that the risk will be calculated for
        (Default: "")

    Returns
    -------
    float
        Percent of records that violate k-anonymity for the given columns (key_vars) in the data frame (df)

    Examples
    --------

    """
    count_df = is_k_anon_col(df, key_vars, k)
    if sensitive_col and label:
        label_df = count_df[count_df[sensitive_col].str.contains(label)]
        s = label_df.shape[0]
        n = s - label_df.is_k_anon.sum()
    else:
        s = count_df.shape[0]
        n = s - count_df.is_k_anon.sum()

    # NOTE: if the label is not found in the column, this will be 0
    if s == 0:
        raise ZeroDivisionError

    return n / s


@BEARTYPE
def is_k_anon(
    df: pd.DataFrame, key_vars: Optional[List[str]] = None, k: int = 5
) -> bool:
    """
    This function takes in a data frame, a k, and list of key variables and
    returns a boolean value for which True if the entire dataframe meets the
    condition for k-anonymity and False if it fails to meet the condition.

    Parameters
    ----------
    df: pd.DataFrame
            A data frame containing records used to calculate k-anonymity

    key_vars : List[str], optional
        A list of the column names from the data frame (df) that will be
        included in the k-anonymity calculation
        (Default: None, all columns will be included)

    k : int, optional
        An integer for k that denotes the threshold for how many unique samples
        can be in a group
        (Default: 5)

    Returns
    -------
    bool
        True if the entire dataframe is k anonymous, False if not

    Examples
    --------

    """
    if key_vars is None:
        key_vars = list(df.columns)
    return all(
        df.groupby(key_vars, observed=True, as_index=False)
        .size()
        .rename(columns={"size": "k_count"})["k_count"]
        > k
    )


@BEARTYPE
def is_k_anon_col(
    df: pd.DataFrame,
    key_vars: Optional[List[str]] = None,
    k: int = 5,
) -> pd.DataFrame:
    """
    This function takes in a data frame, a k, and list of key variables and
    returns the data frame with a boolean column added labeled "is_k_anon"
    which contains True if the row meets the condition for k-anonymity and False
    if it fails to meet the condition.

    Parameters
    ----------
    df: pd.DataFrame
            A data frame containing records used to calculate k-anonymity

    key_vars : List[str], optional
        A list of the column names from the data frame (df) that will be included in the
        k-anonymity calculation
        (Default: None, all columns will be included)

    k : int, optional
        An integer for k that denotes the threshold for how many unique samples can be in a group
        (Default: 5)

    Returns
    -------
    pd.dataframe
        Data frame with k-anonymous column added

    Examples
    --------

    """
    if key_vars is None:
        key_vars = list(df.columns)
    adf = (
        df.groupby(key_vars, observed=True, as_index=False)
        .size()
        .rename(columns={"size": "k_count"})
    )
    adf["is_k_anon"] = adf["k_count"] > k
    return pd.merge(df, adf, on=key_vars)


@BEARTYPE
def _unique_count(q_block: pd.Series) -> float:
    """
    This function counts the number of unique values of the sensitive column in a q-block (a slice of a dataset that
    has the same values of all columns other than the sensitive column). It counts each NA as a single value. This is
    referred to as extended match. See reference.

    Parameters
    ----------
    q_block : pd.Series
        The q-block for which the unique values of the sensitive column are being counted

    Returns
    -------
    float
        The count of the unique values of the sensitive column

    References
    ----------
    https://link.springer.com/content/pdf/10.1007%2F978-3-319-10073-9_27.pdf

    """
    num_unique = q_block.nunique(dropna=True)
    num_na = 1 if sum(q_block.isnull()) > 0 else 0
    return float(num_unique + num_na)


@BEARTYPE
def _entropy_count(q_block: pd.Series) -> float:
    """
    Calculates the entropy of the unique values of the sensitive column in a q-block

    Parameters
    ----------
    q_block : pd.Series
        The q-block for which the entropy of the unique values of the sensitive column are being calculated

    Returns
    -------
    ent: float
        The entropy of the unique values of the sensitive column

    """
    ent = 0
    for L in set(q_block):
        p_l = q_block.value_counts()[L] / len(q_block)
        ent += p_l * np.log(p_l)
    return -ent


@BEARTYPE
def _diversity(
    df: pd.DataFrame,
    sensitive_col: str,
    fxn: Callable[[pd.Series], float],
) -> Dict:
    """
    Creates the q-blocks and applies the l-diversity function to determine the l-diversity of each row and the minimum
    l-diversity of the dataset


    Parameters
    ----------
    df : pd.DataFrame
        A data frame with data that l-diversity is being measured for

    sensitive_col : str
        The name of the column containing the data that is being obscured by mitigations

    fxn : Callable[[pd.DataFrame], float]
        The function being applied to calculate the l-diversity (Example: see _count_unique or _entropy_count above)

    Returns
    -------
    Dict[str, List]
        A dictionary of the l-diversities of every row of the dataset and the minimum l-diversity

    """
    # get the quasi-identifiers
    qi = [colname for colname in df.columns if colname != sensitive_col]
    # group by unique qi values
    grp_qi = df.groupby(qi)
    # get the diversity
    div = grp_qi[sensitive_col].agg(fxn).to_list()
    counts = grp_qi[sensitive_col].agg("count")
    l_div = []
    for dv, ct in zip(div, counts):
        l_div += [dv] * ct

    return {"min": min(div), "l-diversity": l_div}


@BEARTYPE
def l_diversity(
    df: pd.DataFrame,
    sensitive_col: str,
    L: int = 2,
    method: Optional[str] = None,
) -> float:
    """
    Calculates the proportion of the rows with an l-diversity larger than the given `L` parameter.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame with data that l-diversity is being measured for

    sensitive_col : str,
        The name of the column containing the data that is being obscured by mitigations

    L : int, optional
        The threshold by which the closeness of the q-blocks and the full dataset are compared
        (Default: 2)

    method : str {'distinct', 'entropy'}, optional
        The method of l-diversity being applied.
        (Default: 'distinct')

    Returns
    -------
    float
        Percent of records that violate l-diversity

    References
    ----------
    https://en.wikipedia.org/wiki/L-diversity
    https://www.cs.purdue.edu/homes/ninghui/papers/t_closeness_icde07.pdf

    """
    if method is None or method == DISTINCT:
        l_div = _diversity(df, sensitive_col, _unique_count)["l-diversity"]
    elif method == ENTROPY:
        l_div = _diversity(df, sensitive_col, _entropy_count)["l-diversity"]
    else:
        raise ValueError(f"method must be '{DISTINCT}' or '{ENTROPY}'")

    return sum([1.0 if ld <= L else 0.0 for ld in l_div]) / len(l_div)


@BEARTYPE
def is_l_diverse(
    df: pd.DataFrame,
    sensitive_col: str,
    L: int = 2,
    method: Optional[str] = None,
) -> bool:
    """
    Determines if the data set is not l-diverse for a given l value.

    L-diversity is the deterimination that there are l 'well-represented' values for sensitive attribute (column) for
    an equivalence class or q-block. A dataset is considered l-diverse if every q-block is l-diverse.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame with data that l-diversity is being measured for

    sensitive_col : str
        The name of the column containing the data that is being obscured by mitigations

    L : int, optional
        The threshold by which the closeness of the q-blocks and the full dataset are compared. Default is arbitrary.
        (Default: 2)

    method : str {'distinct', 'entropy'}, optional
        The method of l-diversity being applied. From Wikipedia:

            * Distinct l-diversity – The simplest definition ensures that at least L distinct values for the sensitive
            field in each equivalence class exist.

            * Entropy l-diversity – The most complex definition defines Entropy of an equivalent class E to be the
            negation of summation of s across the domain of the sensitive attribute of p(E,s)log(p(E,s)) where p(E,s)
            is the fraction of records in E that have the sensitive value s. A table has entropy l-diversity when for
            every equivalent class E, Entropy(E) ≥ log(L).

        (Default: 'distinct')

    Returns
    -------
    ret: bool
        Boolean indicating if the dataset is l-diverse for a given l value

    References
    ----------
    https://en.wikipedia.org/wiki/L-diversity
    https://www.cs.purdue.edu/homes/ninghui/papers/t_closeness_icde07.pdf

    """
    if method is None or method == DISTINCT:
        return _diversity(df, sensitive_col, _unique_count) <= L
    elif method == ENTROPY:
        return _diversity(df, sensitive_col, _entropy_count) <= np.log(L)

    raise ValueError(f"method must be '{DISTINCT}' or '{ENTROPY}'")


@BEARTYPE
def _get_probs(
    df: pd.DataFrame,
    sensitive_col: str,
) -> Dict[str, int]:
    """
    Given a dataset, calculates the discrete proability of each of the unique values of the sensitive column
    Parameters
    ----------
    df : pd.DataFrame
        A data frame whose sensitive column is being examined
    sensitive_col : str
        The name of the column containing the data that is being obscured by mitigations
    Returns
    -------
    val_probs: Dict[str, int]
        A dictionary with the distinct values of the sensitive column as keys and their probability as values.
    """
    total = float(len(df))
    val_counts = df.groupby(sensitive_col)[sensitive_col].agg("count").to_dict()
    val_probs = {key: value / total for key, value in val_counts.items()}
    return val_probs


@BEARTYPE
def _get_probs_col(col: pd.Series) -> Dict:
    """
    Given a dataset, calculates the discrete proability of each of the unique values of the sensitive column
    Parameters
    ----------
    col : pd.Series
        The pd.Series of the sensitive column that is being examined
    Returns
    -------
    Dict
        A dictionary with the distinct values of the sensitive column as keys and their probability as values.
    """
    total = float(len(col))
    val_counts = col.value_counts().to_dict()
    val_probs = {key: value / total for key, value in val_counts.items()}
    return val_probs


@BEARTYPE
def _dict_diff(dict1: Dict, dict2: Dict) -> List[float]:
    """
    Calculates the absolute difference between the values of two dictionaries by the intersection of their keys. This
    operation is commutable.
    Parameters
    ----------
    dict1 : Dict
        The first dict in the computation. If the dictionaries are of unequal length this must be the shorter.
    dict2 : Dict
        The second dict in the compuation. If the dictionaries are of unequal lenght this must be the longer.
    Returns
    -------
    List[float]
        A list of the differences of the values corresponding with an intersection of the keys in both dictionaries
    """
    dict1_full = {k: (dict1[k] if k in dict1.keys() else 0.0) for k, v in dict2.items()}
    diffs = [float(dict1_full[key]) - float(dict2[key]) for key in dict2.keys()]

    return diffs


@BEARTYPE
def _num_emd(qs: Dict, q_block_col: pd.Series) -> float:
    """
    Calculates the earth mover's distance for numerical values. The earth mover's distance as a metric between
    distributions can be explained as the least amount of work that needs to be done to convert the shape of one
    dirt pile to shape of another, if the distributions are thought of as dirt piles.
    Parameters
    ----------
    qs : Dict
       A dictionary with distinct values of the sensitive column from the full dataset as keys and their probability
       as values.
    q_block_col : pd.Series
        The pd.Series of the sensitive column that is being examined in the q-block
    Returns
    -------
    float
        The earth mover's distance between the distribution of the values of the sensitive column in the q-block and
        the full dataset.

    References
    ----------
    A Tutorial on Computing t-Closeness
    https://arxiv.org/pdf/1911.11212.pdf
    """
    ps = _get_probs_col(q_block_col)
    rs = _dict_diff(ps, qs)
    m = len(rs)
    d_sum = []
    d_sub_sum = 0

    for i in range(1, m):
        for j in range(1, i + 1):
            d_sub_sum += rs[j - 1]
        d_sum += [abs(copy(d_sub_sum))]
        d_sub_sum = 0

    return sum(d_sum) / (m - 1)


@BEARTYPE
def _cat_emd(qs: Dict, q_block_col: pd.Series) -> float:
    """
    Calculates the earth mover's distance for categorical values.
    Parameters
    ----------
    qs : Dict
       A dictionary with distinct values of the sensitive column from the full dataset as keys and their probability
       as values.
    q_block_col : pd.Series
        The pd.Series of the sensitive column that is being examined in the q-block
    Returns
    -------
    float
        The earth mover's distance between the distribution of the values of the sensitive column in the q-block and
        the full dataset.

    References
    ----------
    A Tutorial on Computing t-Closeness
    https://arxiv.org/pdf/1911.11212.pdf

    """
    ps = _get_probs_col(q_block_col)
    rs = _dict_diff(ps, qs)
    rs = [abs(r) for r in rs]
    return sum(rs) / 2.0


@BEARTYPE
def _closeness(
    df: pd.DataFrame,
    sensitive_col: str,
    fxn: Callable[[Dict, pd.DataFrame, str], float],
    qi: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculates the t-closeness of the dataset using the function provided. The provided function determines if the
    the t-closeness is calculated for numerical or categorical values in the sensitive column.
    Parameters
    ----------
    df : pd.DataFrame
        A data frame with data that t-closeness is being measured for
    sensitive_col : str
        The name of only one column containing the data that is being obscured by mitigations
    fxn : Callable[[Dict, pd.DataFrame, str], float]
        The function to be used for numerical or categorical values of the sensitive column (Example: see _cat_emd and
        _num_emd above)
    qi: List[str], optional
        Columns to be considered as the quasi-indicators. If not provided it is assumed that all columns
        but the sensitive column are the quasi-indicators.
        (Default: None)

    Returns
    -------
    Dict[str, float]
        Maximum and overall t-closeness of the q-blocks in the dataset.
    """
    # get the quasi-identifiers
    qi = (
        [colname for colname in df.columns if colname != sensitive_col]
        if qi is None
        else qi
    )
    # group by unique qi values
    grp_qi = df.groupby(qi)
    # get the closeness
    qs = _get_probs(df, sensitive_col)
    fun = lambda x: fxn(qs, x)
    div = grp_qi[sensitive_col].agg(fun)
    counts = grp_qi[sensitive_col].agg("count")

    _t_closeness = []
    for dv, ct in zip(div, counts):
        _t_closeness += [dv] * ct
    return {"max": div.max(), "t-closeness": _t_closeness}


@BEARTYPE
def t_closeness(
    df: pd.DataFrame,
    sensitive_col: str,
    t: float = 0.1,
    datatype: Optional[str] = None,
    qi: Optional[List[str]] = None,
    test: bool = False,
) -> Union[List, float]:
    """
    Determines if the t-closeness is more than the parameter t
    Parameters
    ----------
    df : pd.DataFrame
        A data frame with data that t-closeness is being measured for
    sensitive_col : str
        The name of only one column containing the data that is being obscured by mitigations
    t : float [0.0, 1.0], optional
        The threshold by which the closeness of the q-blocks and the full dataset are compared
        (Default: 0.1)
    datatype : str {'categorical', 'numeric'}, optional
        The datatype of the sensitive column, must be either 'categorical' or 'numeric'
        (Default: 'categorical')
    qi: List[str], optional
        Columns to be considered as the quasi-indicators. If not provided it is assumed that all columns
        but the sensitive column are the quasi-indicators.
        (Default: None)
    test: bool, optional
        If true function returns list of closeness measurements (for testing only)

    Returns
    -------
    float
        The proportion of the rows that are closer than the t threshold
    """
    if isinstance(sensitive_col, list):
        raise ValueError("sensitive column must be a single string")

    if datatype is None or datatype == NUMERIC:
        t_cls = _closeness(df, sensitive_col, _num_emd, qi)["t-closeness"]
    elif datatype == CATEGORICAL:
        t_cls = _closeness(df, sensitive_col, _cat_emd, qi)["t-closeness"]
    else:
        raise ValueError(f"datatype must be '{NUMERIC}' or '{CATEGORICAL}'")

    if test:
        return t_cls

    return sum([1.0 if tc >= t else 0.0 for tc in t_cls]) / len(t_cls)


@BEARTYPE
def is_t_close(
    df: pd.DataFrame,
    sensitive_col: str,
    t: float = 0.1,
    datatype: str = "categorical",
    qi: Optional[List[str]] = None,
) -> bool:
    """
    Determines if the t-closeness is more than the parameter t
    Parameters
    ----------
    df : pd.DataFrame
        A data frame with data that t-closeness is being measured for
    sensitive_col : str
        The name of only one column containing the data that is being obscured by mitigations
    t : float [0.0, 1.0], optional
        The threshold by which the closeness of the q-blocks and the full dataset are compared. Default is arbitrary.
        (Default: 0.1)
    datatype : str {'categorical', 'numeric'}, optional
        The datatype of the senstive column, must be either 'categorical' or 'numeric'
        (Default: 'categorical')
    qi: List[str], optional
        Columns to be considered as the quasi-indicators. If not provided it is assumed that all columns
        but the sensitive column are the quasi-indicators.
        (Default: None)

    Returns
    -------
    ret: bool
        Whether the dataset is t-close or not.
    """
    if isinstance(sensitive_col, list):
        raise ValueError("sensitive column must be a single string")

    if datatype is None or datatype == CATEGORICAL:
        return _closeness(df, sensitive_col, _cat_emd, qi)["max"] >= t
    if datatype == NUMERIC:
        return _closeness(df, sensitive_col, _num_emd, qi)["max"] >= t

    raise ValueError(f"datatype must be '{NUMERIC}' or '{CATEGORICAL}'")


@BEARTYPE
def indiv_risk_approx(fk: Union[int, float], Fk: Union[int, float]) -> float:
    """
    calculates the approximate individual risk

    Parameters
    ----------
    fk : int or float
        the sample frequency of the row's combination of quasi-identifier values
    Fk : int or float
        the population frequence of the row's combination of quasi-identifier values

    Returns
    -------
    float
        risk score in [0, 1]

    Examples
    --------
    TODO

    """
    if fk == Fk:
        return 1 / float(fk)

    pk = float(fk) / float(Fk)

    if fk > 2:
        return pk / (fk - (1 - pk))
    if fk == 2:
        return (pk / (1 - pk)) - (((pk / (1 - pk)) ^ 2) * np.log(1 / pk))
    return (pk / (1 - pk)) * np.log(1 / pk)


@BEARTYPE
def indiv_risk_exact(fk: int, Fk: float) -> float:
    """
    calculates the exact individual risk

    Parameters
    ----------
    fk : int
        the sample frequency of the row's combination of quasi-identifier values
    Fk : int
        the population frequence of the row's combination of quasi-identifier values

    Returns
    -------
    float
        risk score in [0, 1]

    Examples
    --------
    TODO

    """
    if fk == Fk:
        return 1 / float(fk)

    pk = float(fk) / float(Fk)

    def B(fk, pk, i):
        b1 = (fk - 1 - i) ^ 2 / ((i + 2) * (fk - 2 - i))
        b2 = (pk ^ (i + 2 - fk) - 1) / (pk ^ (i + 1 - fk) - 1)
        return b1 * b2

    def BB(fk, pk):
        bb = 0
        for m in range(fk - 2):
            b = 1
            for m2 in range(m + 1):
                b = b * B(fk, pk, m2)
            bb = bb + (-1) ^ (m + 1) * b
        return bb

    first = (pk / (1 - pk)) ^ fk
    third = (-1) ^ fk * np.log(pk)

    if fk > 2:
        A = (pk ^ (1 - fk) - 1) / (fk - 1)
        return first * ((A * (1 + BB(fk, pk))) + third)
    if fk == 2:
        return (pk / (1 - pk)) - (((pk / (1 - pk)) ^ 2) * np.log(1 / pk))
    return (pk / (1 - pk)) * np.log(1 / pk)


@formatting(on_output=False, ignore_dtypes=True)
@BEARTYPE
def indiv_risk(
    df: pd.DataFrame,
    sensitive_col: Union[int, str],
    quasi_cols: List[Union[int, str]],
    method: str = "approx",
    weights: Optional[List[Union[int, float]]] = None,
    qual: Union[int, float] = 1,
):
    """Calculate the individual risk each value contributes in re-identifying sensitve variable.

    Parameters
    ----------
    df : DataFrame, Series, or array_like
        The data to be modified.
    sensitive_col : str or int
        Column whos relationship to other columns needs to be obfuscated.
    quasi_cols : list
        Columns to consider when calculating individual risk of re-identifying `sensitive_col`.
    method : {'approx', 'exact'} (Default: 'approx')
        Precision for calculating the individual risk of each value.
    weights: list, optional (Default: None)
        Proportion of rows with the same combination of values in `data` for
        each row in `data`, only to be used if `data` is a sample/subset of a larger dataset.
        If None, `weights` will be a unit vector indicating that `data` is the full population.
    qual: float (0, 1], optional (Default: 1)
        Perceived quality of frequency counts to act as a final correction factor.

    Returns
    -------
    DataFrame
        A DataFrame with the individual risk values of each row.

    Raises
    ------
    InputError:
        Raised when invalid arguments are passed.
    NotInRangeError
        Raised if `risk_threshold` is outside the interval [0, 1].
    """
    if not (0 < qual <= 1):
        raise NotInRangeError("Qual is out of bounds: [0 < qual <= 1]")

    if not all(col in df.columns for col in [sensitive_col] + quasi_cols):
        raise InputError(
            f"Values of `sensitive_col` and/or `quasi_cols` are not in `data` columns {list(df.columns)}. (Received: {sensitive_col} and {quasi_cols})"
        )

    df["order"] = df.index

    freq_count = freq_calc(
        df,
        sensitive_col=sensitive_col,
        quasi_cols=quasi_cols,
    )

    if method == APPROX:
        freq_count["risk"] = freq_count.apply(
            lambda x: indiv_risk_approx(x["samp_fq"], x["pop_fq"]) * qual, axis=1
        )
    elif method == EXACT:
        freq_count["risk"] = freq_count.apply(
            lambda x: indiv_risk_exact(x["samp_fq"], x["pop_fq"]) * qual, axis=1
        )
    else:
        raise InputError(
            f"Method must be in ['{APPROX}', '{EXACT}'] Method given was {method}"
        )

    return pd.merge(df, freq_count, how="left", on=quasi_cols + ["order"])["risk"]


@BEARTYPE
def beta_likeness(
    df: pd.DataFrame,
    sensitive_col: str,
    beta: Union[int, float] = 1.0,
    qi: Optional[List] = None,
    enhanced: bool = True,
    bool_return: bool = False,
) -> Union[bool, float]:
    """
    Determines the percent of entries in the data that do not satisfy beta-likeness. If
    bool_return is set to True, returns True if the entire dataset satisfies beta-likeness for a
    given beta value, False otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame with data that beta-likeness is being measured for
    sensitive_col : str
        The name of only one column containing the data that is being obscured by mitigations
    beta : float (>0), optional
        The tolerance threshold for the increase in confidence of a certain sensitive attribute,
        in relative difference terms, for SAs in an equivalence class versus the total population
        (Default: 1.0)
    qi : List, optional
        Columns to be considered as the quasi-indicators. If not provided it is assumed that all columns
        but the sensitive column are the quasi-indicators.
        (Default: None)
    enhanced : bool, optional
        Should the function run enhanced beta-likeness instead of basic beta-likeness.
        (Default: True)
    bool_return : bool, optional
        Should the function return a boolean indicating True if beta-likeness is satisfied/False
        if beta-likeness is not satisfied.
        (Default: False)

    Returns
    -------
    bool
        If the `bool_return` flag is set to True, the function will return True if the supplied dataframe
        satisfies beta likeness for the supplied sensitive column and beta value
    float
        The proportion of the rows that fail beta-likeness

    Raises
    ------
    InputError
        This error is raised when a `beta` value of <= 0 is supplied.
    """
    if not beta > 0:
        raise InputError("beta must be a value greater than 0")
    qi = (  # Generate a list of all quasi-indicators (qi)
        [colname for colname in df.columns if colname != sensitive_col]
        if qi is None
        else qi
    )

    # group by unique qi values
    grp_qi = df.groupby(qi)

    # get the frequency of SA values in the full dataset
    sa_all = _get_probs(df, sensitive_col)
    # V (SA values) = sa_all keys
    # P (SA probabilities) = sa_all values

    failed_row_ct = 0

    for key, item in grp_qi:
        sa_ec = _get_probs(
            item, sensitive_col
        )  # get the frequencies of SA values in the equivalence class
        for key in sa_ec.keys():
            if not sa_all[key] < sa_ec[key]:  # satisfies the requirement that p_i < q_i
                continue
            dist = (sa_ec[key] - sa_all[key]) / sa_all[key]  # (q_i - p_i) / p_i
            if enhanced:
                threshold = np.minimum(beta, -np.log(sa_all[key]))
            else:
                threshold = beta
            if dist > threshold:
                if bool_return:
                    return False
                failed_row_ct += len(item[item[sensitive_col] == key])

    if bool_return:
        return True
    return failed_row_ct / len(df)


@BEARTYPE
def is_beta_like(
    df: pd.DataFrame,
    sensitive_col: str,
    beta: float = 1.0,
    qi: Optional[List[str]] = None,
    enhanced: bool = True,
) -> bool:
    """
    Determines if the beta-likeness is satisfied for parameter beta

    Parameters
    ----------
    df : pd.DataFrame
        A data frame with data that beta-likeness is being measured for
    sensitive_col : str
        The name of only one column containing the data that is being obscured by mitigations
    beta : float (>0), optional
        The tolerance threshold for the increase in confidence of a certain sensitive attribute,
        in relative difference terms, for SAs in an equivalence class versus the total population
        (Default: 1.0)
    qi : List[str], optional
        Columns to be considered as the quasi-indicators. If not provided it is assumed that all columns
        but the sensitive column are the quasi-indicators.
        (Default: None)
    enhanced : bool, optional
        Should the function run enhanced beta-likeness instead of basic beta-likeness.
        (Default: True)

    Returns
    -------
    ret: bool
        Whether the dataset satisfies beta-likeness or not.
    """
    return beta_likeness(df, sensitive_col, beta, qi, enhanced, bool_return=True)
