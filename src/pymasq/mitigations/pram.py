import pandas as pd
import numpy as np

from typing import Dict, List, Optional, Union

from pymasq import BEARTYPE
from pymasq.config import (
    FORMATTING_ON_OUTPUT,
    FORMATTING_IGNORE_DTYPES,
)
from pymasq.errors import InputError, NotInRangeError
from pymasq.mitigations.utils import _is_identical
from pymasq.utils import formatting


__all__ = ["pram"]


def __calc_transition_matrix(
    data: pd.Series,
    cats: pd.Categorical,
    probs: Union[int, float],
    alpha: Union[int, float],
) -> pd.DataFrame:
    """Calculate the transition matrix between categorical values

    Parameters
    ----------
    data : pandas.Series
        Data to be perturbed.
    cats : pandas.Categorical
        Allowed categorical values for `data`.
    probs : int or float
        Probabilities to use when calculating the transition matrix.
    alpha : int or float
        Perturbation magnitude.

    Returns
    -------
        pandas.DataFrame with transition probabilities for each category.
    """
    ncats = len(cats)
    runif = np.random.uniform(low=probs, size=ncats)
    tri = (1 - runif) / (ncats - 1)

    prob_mat = np.zeros(shape=(ncats, ncats))
    prob_mat[:, :] = tri.reshape((1, ncats)).T
    np.fill_diagonal(prob_mat, runif)

    cat_codes = data.cat.codes + 1
    sum_cats = np.nansum(cat_codes)
    freqs = data.value_counts() / sum_cats  # scaled category frequencies

    scaled_prob_mat = prob_mat.copy()
    for i in range(ncats):
        s = sum(freqs * prob_mat[:, i])
        for j in range(ncats):
            scaled_prob_mat[i, j] = prob_mat[j, i] * (freqs[j] / s)

    trans_probs = prob_mat @ scaled_prob_mat
    scaled_trans_probs = alpha * trans_probs + (1 - alpha) * np.identity(ncats)

    return pd.DataFrame(scaled_trans_probs, index=cats, columns=cats)


def __randomization(
    data: pd.Series, trans: pd.DataFrame, cats: pd.Categorical
) -> pd.DataFrame:
    """Perform PRAM on a pandas Series.

    Randomly change the category of an entry based on the transition probabilities.

    Parameters
    ----------
    data : pandas.Series
        Data to be perturbed.
    trans : pandas.DataFrame
        Probabilities for transitioning between category values for each category.
    cats : pandas.Categorical
        Allowed categorical values for `data`.

    Returns
    -------
    pd.DataFrame
        A DataFrame with perturbed values.
    """
    d_pramed = data.copy()
    d_pramed.cat.set_categories(cats)
    for cat in cats:
        idxs = data.index.where(data == cat).dropna()
        if len(idxs) > 0:
            d_pramed[idxs] = np.random.choice(
                cats,
                len(idxs),
                p=trans.loc[
                    cat,
                ],
            )

    return d_pramed


@formatting(on_output=FORMATTING_ON_OUTPUT)
@BEARTYPE
def pram(
    data: Union[pd.DataFrame, pd.Series],
    perturb_cols: Optional[List] = None,
    sensitive_col: Optional[str] = None,
    probs: Union[float, int, Dict] = 0.8,
    alpha: Union[float, int] = 0.5,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
):
    """
    The algorithm randomly changes the values of variables in selected records (usually the risky ones) according
    to an invariant probability transition matrix or a custom-defined transition matrix.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be modified.
    perturb_cols : List, Optional (Default: None)
        The subset of columns that will be manipulated from `data`. The values of `data[perturb_cols]`
        must be categorical, or able to be cast as categorical, and there can be no overlapping columns
        with `sensitive_col`. If omitted, all columns of `data` will be perturbed.
    sensitive_col : str (Default: None)
        Column whos relationship to other columns needs to be obfuscated.
        If defined, then only the columns specified in `perturb_cols` will be modified.
        There can be no overlapping columns with `perturb_cols`.
    probs: float, int, dict (Default: 0.8)
        Probabilities to use when calculating the transition matrix. If `probs` is a float or int,
        then it must be between the interval [0,1] and it will be used as a lower bound probability
        that a categorical value will *not* change. Thus, if `probs` is 1, then there will *not* be
        any change in the categorical values, and if `probs` is 0, then there is a high probability
        that a categorical value will change. If `probs` is a dict, then it should contain the
        transition probabilities for each column in `perturb_cols`, where the keys specify the
        column name and the value will be the transition probabilities specified as a pandas DataFrame
        of size (N, N), where N is the number of categories in the column as returned by a single
        column or `pandas.Series.cat.categories`. The index and column names of the transition probabilities
        must be set to the available categorical values. Note that the horizontal sum of each of the
        transition probabilities should be equal to 1.
    alpha : float or int (Default: 0.5)
        Perturbation magnitude. This parameter scales the randomly generated transition probabilities.
        Must be between the interval [0,1], where 0 indicates *no* categorical values of `data` will be changed
        and 1 indicates a high probability of change in categorical values since the computed transition probabilities not be scaled down.
        `alpha` is disregarded if the transition probabilities, `probs`, is provided as a Dict.
    cols : str or list (Default: None)
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with PRAMed values.

    Raises
    ------
    InputError:
        Raised when invalid arguments are passed.
    NotInRangeError
        Raised if `alpha` or `probs` (if `probs` is passed as an integer or float) are outside the interval [0, 1].

    Examples
    --------
    >>> df = pymasq.datasets.load_census()
    >>> df = df[['workclass', 'education', 'relationship', 'race', 'sex']].head(10)
    >>> df
              workclass  education   relationship   race     sex
    0         State-gov  Bachelors  Not-in-family  White    Male
    1  Self-emp-not-inc  Bachelors        Husband  White    Male
    2           Private    HS-grad  Not-in-family  White    Male
    3           Private       11th        Husband  Black    Male
    4           Private  Bachelors           Wife  Black  Female
    5           Private    Masters           Wife  White  Female
    6           Private        9th  Not-in-family  Black  Female
    7  Self-emp-not-inc    HS-grad        Husband  White    Male
    8           Private    Masters  Not-in-family  White  Female
    9           Private  Bachelors        Husband  White    Male


    >>> pram(df)
              workclass  education   relationship   race     sex
    0         State-gov  Bachelors  Not-in-family  White    Male
    1  Self-emp-not-inc  Bachelors        Husband  White  Female
    2           Private    HS-grad  Not-in-family  White    Male
    3           Private        9th        Husband  Black    Male
    4           Private  Bachelors  Not-in-family  White  Female
    5           Private    Masters           Wife  White  Female
    6           Private        9th  Not-in-family  Black  Female
    7  Self-emp-not-inc       11th        Husband  White    Male
    8           Private    Masters  Not-in-family  White  Female
    9           Private  Bachelors        Husband  Black    Male

    >>> pram(df, perturb_cols=["workclass"])
              workclass  education   relationship   race     sex
    0         State-gov  Bachelors  Not-in-family  White    Male
    1  Self-emp-not-inc  Bachelors        Husband  White    Male
    2  Self-emp-not-inc    HS-grad  Not-in-family  White    Male
    3           Private       11th        Husband  Black    Male
    4           Private  Bachelors           Wife  Black  Female
    5           Private    Masters           Wife  White  Female
    6           Private        9th  Not-in-family  Black  Female
    7  Self-emp-not-inc    HS-grad        Husband  White    Male
    8           Private    Masters  Not-in-family  White  Female
    9  Self-emp-not-inc  Bachelors        Husband  White    Male

    >>> pram(df, perturb_cols=["workclass"], sensitive_col="sex")
              workclass  education   relationship   race     sex
    0           Private  Bachelors  Not-in-family  White    Male
    1  Self-emp-not-inc  Bachelors        Husband  White    Male
    2           Private    HS-grad  Not-in-family  White    Male
    3           Private       11th        Husband  Black    Male
    4           Private  Bachelors           Wife  Black  Female
    5           Private    Masters           Wife  White  Female
    6           Private        9th  Not-in-family  Black  Female
    7  Self-emp-not-inc    HS-grad        Husband  White    Male
    8           Private    Masters  Not-in-family  White  Female
    9           Private  Bachelors        Husband  White    Male

    >>> pram(df, perturb_cols=["workclass", "education","relationship"], sensitive_col="sex", probs=0.2, alpha=0.5)
              workclass  education   relationship   race     sex
    0           Private  Bachelors        Husband  White    Male
    1           Private       11th        Husband  White    Male
    2           Private    HS-grad  Not-in-family  White    Male
    3           Private       11th        Husband  Black    Male
    4  Self-emp-not-inc    HS-grad           Wife  Black  Female
    5           Private    Masters           Wife  White  Female
    6           Private    Masters  Not-in-family  Black  Female
    7  Self-emp-not-inc  Bachelors        Husband  White    Male
    8           Private        9th  Not-in-family  White  Female
    9           Private  Bachelors        Husband  White    Male

    >>> pram(df, sensitive_col="sex", alpha=0.8)
              workclass  education   relationship   race     sex
    0         State-gov       11th  Not-in-family  White    Male
    1  Self-emp-not-inc  Bachelors           Wife  White    Male
    2           Private    HS-grad  Not-in-family  White    Male
    3           Private       11th        Husband  Black    Male
    4           Private  Bachelors        Husband  Black  Female
    5           Private        9th           Wife  White  Female
    6           Private        9th  Not-in-family  Black  Female
    7  Self-emp-not-inc    HS-grad        Husband  Black    Male
    8           Private    Masters  Not-in-family  White  Female
    9  Self-emp-not-inc  Bachelors        Husband  White    Male

    >>> # define transition probabilities for changing categories in the race column
    >>> probs = dict(race=pd.DataFrame({"White": 0.5, "Black": 0.5}, index=["White", "Black"]))
    >>> probs
           White  Black
    White    0.5    0.5
    Black    0.5    0.5

    >>> pram(df, perturb_cols=["race"], probs=probs, sensitive_col="sex")
              workclass  education   relationship   race     sex
    0         State-gov  Bachelors  Not-in-family  Black    Male
    1  Self-emp-not-inc  Bachelors        Husband  White    Male
    2           Private    HS-grad  Not-in-family  White    Male
    3           Private       11th        Husband  Black    Male
    4           Private  Bachelors           Wife  Black  Female
    5           Private    Masters           Wife  White  Female
    6           Private        9th  Not-in-family  White  Female
    7  Self-emp-not-inc    HS-grad        Husband  Black    Male
    8           Private    Masters  Not-in-family  Black  Female
    9           Private  Bachelors        Husband  Black    Male
    """
    try:
        data = data.astype("category")
    except:
        raise InputError(
            f"All values of `data` must be of type categorical and cannot be coerced. (Received {data.dtypes})"
        )

    if perturb_cols is None:
        perturb_cols = [col for col in data.columns if col != sensitive_col]
    [sensitive_col] if sensitive_col is not None else []
    if not all(
        [
            col in data.columns
            for col in perturb_cols
            + ([sensitive_col] if sensitive_col is not None else [])
        ]
    ):
        raise InputError(
            f"Values of `perturb_cols` and/or `sensitive_col` are not in `data` columns: {list(data.columns)}. (Received: {perturb_cols} and {sensitive_col}, respectively)"
        )

    if set(perturb_cols).intersection([sensitive_col]):
        raise InputError(
            f"Values of `perturb_cols ` and `sensitive_col` must not overlap. (Received: {perturb_cols} and {sensitive_col}, respectively)"
        )

    n_pc = len(perturb_cols)
    perturb_cols = [
        c
        for c in perturb_cols
        if not data[c].isnull().all() and not _is_identical(data[c])
    ]

    if len(perturb_cols) != n_pc:
        if len(perturb_cols) == 0:
            raise InputError(f"All values of `data` cannot be NaNs or identical.")
        else:
            print(
                "WARNING: ignoring columns that are composed entirely of identical values."
            )

    if not 0.0 <= alpha <= 1.0:
        raise NotInRangeError(
            f"Perturbation magnitude `alpha` must be between interval [0,1], inclusive. (Received: {alpha})"
        )

    if isinstance(probs, (int, float)) and not 0.0 <= probs <= 1.0:
        raise NotInRangeError(
            f"Perturbation probabilities `probs` must be between interval [0,1], inclusive, if it is passed as an int or float. (Received: {probs})"
        )

    tmp_col = "tmp_factors"
    if sensitive_col is not None:
        data[tmp_col] = data[sensitive_col]
    else:
        data[tmp_col] = np.ones(len(data))

    sens_cats = data[tmp_col].unique()

    for col in perturb_cols:
        series = data[col]
        cats = series.cat.categories

        if isinstance(probs, dict):
            try:
                transmat = probs[series.name]
            except:
                raise InputError(
                    f"Invalid key in `probs` dict ({list(probs.keys())}). ensure that all `perturb_cols` are present as keys in `probs` dict. (Received {series.name})"
                )
        else:
            transmat = __calc_transition_matrix(series, cats, probs, alpha)

        for sens_cat in sens_cats:
            idxs = series.notna().index.where(data[tmp_col] == sens_cat).dropna()
            perturbed = __randomization(series[idxs], transmat, cats=cats)
            data.loc[perturbed.index, col] = perturbed.values

    data = data.drop(columns=tmp_col)

    return data
