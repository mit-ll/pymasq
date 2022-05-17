import math
from typing import Union, List, Final, Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import scipy.stats as ss

from pymasq import BEARTYPE
from pymasq.config import FORMATTING_ON_OUTPUT
from pymasq.utils import formatting
from pymasq.preprocessing import LabelEncoder_pm
from pymasq.errors import InputError, DataTypeError
from pymasq.mitigations.utils import _is_identical


__all__ = [
    "shuffle",
    # "shuffle_by_correlation", # TODO: add only if shuffle_by_model will also be added (replaces current shuffle)
    # "shuffle_by_model", # TODO
    "SPEARMAN",
    "PEARSON",
    "KENDALL",
    "CORRELATIVE",
    "MODEL",
]


SPEARMAN: Final = "spearman"
PEARSON: Final = "pearson"
KENDALL: Final = "kendall"
CORRELATIVE: Final = "corr"
MODEL: Final = "model"


@BEARTYPE
def _reverse_map(data: pd.DataFrame, y_star: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in two dataframes performs data and remaps the data based on
    the order of the columns and returns a reorder dataframe

    Parameters
    ----------
    data : pd.DataFrame
        A data frame that contains the original, unaltered data

    y_star : pd.DataFrame
        A data frame with randomly shuffled data based on correlation
        with the original data

    Returns
    -------
    y_star : pd.DataFrame
        A dataframe containing the shuffled data.
    """
    x_idx_name = data.index.name
    y_idx_name = y_star.index.name

    if not x_idx_name:
        name = "index"
        data.index.set_names(name, inplace=True)
        x_idx_name = name
        y_star.index.set_names(name, inplace=True)
        y_idx_name = name

    for col in y_star.columns:
        y_idx = y_star[col].rank().sort_values().index
        x_idx = data[col].rank().sort_values().index
        x = data.loc[x_idx, col].reset_index()
        y = data.loc[y_idx, col].reset_index()
        x[col] = y[col]
        x.set_index(x_idx_name, inplace=True)
        if isinstance(x[col], pd.Series):
            data[col] = x
        else:
            data[col] = x[col]
        x_idx = data[col].rank().sort_values().index
        x = data.loc[x_idx, col].reset_index()
        y = y_star.loc[y_idx, col].reset_index()
        y[col] = x[col]
        y.set_index(y_idx_name, inplace=True)
        y_star[col] = y[col]

    return y_star


def shuffle_by_model(
    data: pd.DataFrame,
    pred_cols: Optional[List[str]] = None,
    cols: Optional[Union[str, List[str]]] = None,
    seed: Optional[int] = None,
) -> pd.Series:
    """"""
    # TODO Write Model Function
    raise NotImplementedError


@formatting(on_output=FORMATTING_ON_OUTPUT)
@BEARTYPE
def shuffle(
    data: pd.DataFrame,
    shuffle_cols: List[Union[str, int]],
    cor_cols: Optional[List[Union[str, int]]] = None,
    cor_method: str = "spearman",
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
):
    """Shuffle numeric columns based on correlation to other columns.

    The shuffe technique aim to mix up data and can optionally retain
    logical relationships between columns. It randomly shuffles data from
    a dataset within an attribute (e.g. a column in a pure flat format) or
    a set of attributes (e.g. a set of columns). You can shuffle sensitive
    information to replace it with other values for the same attribute from
    a different record.

    Parameters
    ----------
    data : DataFrame or array_like
        The dataframe of numeric columns to be modified.
    shuffle_cols : List of str or ints
        The subset of columns that will be shuffled from `data`. The values of `data[shuffle_cols]`
        must be numeric and there can be no overlapping columns with `cor_cols`.
    cor_cols : List of str or ints, Optional (Default: None)
        The subset of columns used to calculate correlation with `shuffle_cols`. If not set, then all columns
        of `data` not specified in `shuffle_cols` will be used. There can be no overlapping columns with `shuffle_cols`.
    cor_method : {'spearman', 'pearson', 'kendall'}, Optional (Default: 'spearman')
        Method used to generate covariance
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Returns
    -------
    pd.DataFrame
        The shuffled DataFrame for the specified column(s).

    Examples
    --------
    >>> df = pymasq.datasets.load_loan().dropna().reset_index()
    >>> df = df[['ApplicantIncome', 'LoanAmount', 'Married', 'Education', 'Loan_Status']].head(10)
    >>> df
        ApplicantIncome  LoanAmount  Married  Education     Loan_Status
    0   4583             128.0       Yes      Graduate      N
    1   3000             66.0        Yes      Graduate      Y
    2   2583             120.0       Yes      Not Graduate  Y
    3   6000             141.0       No       Graduate      Y
    4   5417             267.0       Yes      Graduate      Y
    5   2333             95.0        Yes      Not Graduate  Y
    6   3036             158.0       Yes      Graduate      N
    7   4006             168.0       Yes      Graduate      Y
    8   12841            349.0       Yes      Graduate      N
    9   3200             70.0        Yes      Graduate      Y

    >>> shuffle(df, shuffle_cols=["ApplicantIncome"])
        ApplicantIncome  LoanAmount  Married  Education     Loan_Status
    0   4583             128.0       Yes      Graduate      N
    1   3000             66.0        Yes      Graduate      Y
    2   2583             120.0       Yes      Not Graduate  Y
    3   4006             141.0       No       Graduate      Y
    4   5417             267.0       Yes      Graduate      Y
    5   2333             95.0        Yes      Not Graduate  Y
    6   6000             158.0       Yes      Graduate      N
    7   3200             168.0       Yes      Graduate      Y
    8   12841            349.0       Yes      Graduate      N
    9   3036             70.0        Yes      Graduate      Y

    >>> shuffle(df, shuffle_cols=["ApplicantIncome", "LoanAmount"], cor_cols=["Education", "Loan_Status"])
        ApplicantIncome  LoanAmount  Married  Education     Loan_Status
    0   4006             70.0        Yes      Graduate      N
    1   5417             158.0       Yes      Graduate      Y
    2   2333             66.0        Yes      Not Graduate  Y
    3   6000             128.0       No       Graduate      Y
    4   3200             141.0       Yes      Graduate      Y
    5   2583             349.0       Yes      Not Graduate  Y
    6   3000             120.0       Yes      Graduate      N
    7   3036             95.0        Yes      Graduate      Y
    8   12841            267.0       Yes      Graduate      N
    9   4583             168.0       Yes      Graduate      Y
    """
    if cor_cols is None:
        cor_cols = [col for col in data.columns if col not in shuffle_cols]

    if not all([col in data.columns for col in [*shuffle_cols, *cor_cols]]):
        raise InputError(
            f"Values of `shuffle_cols` and/or `cor_cols` are not in `data` columns: {list(data.columns)}. (Received: {shuffle_cols} and {cor_cols}, respectively)"
        )

    if set(shuffle_cols).intersection(cor_cols):
        raise InputError(
            f"Values of `shuffle_cols` and `cor_cols` must not overlap. (Received: {shuffle_cols} and {cor_cols}, respectively)"
        )

    y = data[shuffle_cols]
    x = data[cor_cols]

    if not is_numeric_dtype(y.values):
        raise InputError(
            f"Columns to be shuffled can only be numeric. (Received: {shuffle_cols})"
        )

    ignored_cols = [
        (i, x.pop(c)) for i, c in enumerate(x.columns) if _is_identical(x[c])
    ]
    if len(ignored_cols) > 0:
        if len(x.columns) == 0:
            raise InputError(
                f"The values of `data[{cor_cols}]` are all identical and therefore cannot be used for correlation."
            )
        else:
            print(
                "WARNING: ignoring columns that are composed entirely of identical values."
            )

    _data = LabelEncoder_pm.encode(df=pd.concat([x, y], axis=1))

    resp_cols = y.columns.to_list()
    pred_cols = x.columns.to_list()

    ranks = _data.rank()
    perc = (ranks - 0.5) / _data.shape[0]
    norm_inv = pd.DataFrame(ss.norm.ppf(perc), columns=_data.columns)
    pmc = 2 * np.sin(_data.corr(method=cor_method) * math.pi / 6)
    pxs = pmc.loc[resp_cols, pred_cols]
    pxx = pmc.loc[resp_cols, resp_cols]
    psx = pmc.loc[pred_cols, resp_cols]
    pssinv = np.linalg.inv(pmc.loc[pred_cols, pred_cols])
    pssinv = pd.DataFrame(pssinv, columns=pred_cols, index=pred_cols)
    predictors = norm_inv.loc[:, pred_cols]
    ystar1 = predictors.dot(pxs.dot(pssinv).T)

    sigma = pxx - pxs.dot(pssinv.dot(psx))
    e1 = np.random.multivariate_normal(
        mean=[0] * len(resp_cols), cov=sigma, size=_data.shape[0]
    )
    y_star = ystar1 + e1
    y_star.set_index(_data.index, inplace=True)

    data[resp_cols] = _reverse_map(_data, y_star)

    return data


'''
### Not currently being used; keeping for future integration 
def _shuffle_wrapper(
    data: pd.DataFrame,
    method: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Takes in a dataframe performs data shuffling on numeric columns
    and returns the dataframe.

    Wrapper function for `shuffle_by_correlation` and `shuffle_by_model`.

    Shuffling techniques aim to mix up data and can optionally retain
    logical relationships between columns. It randomly shuffles data from
    a dataset within an attribute (e.g. a column in a pure flat format) or
    a set of attributes (e.g. a set of columns). You can shuffle sensitive
    information to replace it with other values for the same attribute from
    a different record.

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The numeric data to be modified.

    method : {"corr", "model"} (Default: 'corr')
        The method used to generate the predictions for shuffling

    **kwargs
        Additional arguments to be passed to `shuffle_by_model` and `shuffle_by_correlation`.

            If `method` is "model":


            If `method` is "corr":

                * cor_method : str, {'spearman','kendall','pearson'}, optional
                    Method used to generate covariance

                * test : bool, optional
                    Flag for whether the shuffle is a test, if True then Ystar1 without the
                    random error is returned for reproducibility
                    (Default: False)

    Returns
    -------
    DataFrame
        A DataFrame with shuffled values.

    Raises
    ------
    InputError
        This error is raised when the parameters supplied for a given method do not match the
        appropriate type of the expected parameters of that method.

    InputError
        This error is raised when a `method` is provided either without the required parameters for
        that `method`.

    See Also
    --------
    pymasq.mitigations.shuffle_by_model : Shuffles all the values using a linear model to generate predictions.

    pymasq.mitigations.shuffle_by_correlation : Shuffles all the values using correlated noise to generate predictions.

    Examples
    --------
    >>> TODO
    """
    if method == CORRELATIVE:
        return shuffle(
            data,
            shuffle_cols=kwargs["shuffle_cols"],
            cor_cols=kwargs.get("cor_cols", None),
            cor_method=kwargs.get("cor_method", SPEARMAN),
        )

    elif method == MODEL:
        # return shuffle_by_model(y, X)
        pass

    raise InputError(
        f"Invalid `method` defined; method must be one of ['model', 'corr']. (Received: {method}"
    )
'''