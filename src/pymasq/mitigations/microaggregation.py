from typing import Any, Callable, Dict, Final, List, Optional, Union

import numpy as np
import pandas as pd

from scipy.stats import shapiro, zscore
from sklearn.cluster import KMeans, DBSCAN, Birch, AgglomerativeClustering, OPTICS
from sklearn.covariance import MinCovDet
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import scale as sklearn_scale
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, FastICA

from pymasq import BEARTYPE
from pymasq.config import (
    FORMATTING_ON_OUTPUT,
    FORMATTING_IGNORE_DTYPES,
    VALIDATE_NUMERIC_ON_INPUT,
    VALIDATE_NUMERIC_ON_OUTPUT,
)
from pymasq.utils import validate_numeric, formatting
from pymasq.errors import InputError, NotInRangeError, LessThanOrEqualToZeroError

import logging

logger = logging.getLogger(__name__)


try:
    from tensorly.decomposition import robust_pca
except ImportError:
    robust_pca = None


__all__ = [
    "quantile_magg",
    "sequential_magg",
    "individual_ranking_magg",
    "robust_magg",
    "advanced_magg",
    "microaggregation",
    "MaggMethods",
    "MaggConst",
    "MaggClusteringMethods",
    "MaggScalingMethods",
    "MaggReductionMethods",
]


class MaggMethods:
    ADVANCED: Final = "advanced"
    QUANTILE: Final = "quantile"
    RANKING: Final = "ranking"
    ROBUST: Final = "robust"
    SEQUENTIAL: Final = "sequential"


class MaggConst:
    DROP: Final = "drop"
    RAISE: Final = "raise"


class MaggClusteringMethods:
    BIRCH: Final = "birch"
    KMEANS: Final = "kmeans"
    DBSCAN: Final = "dbscan"
    OPTICS: Final = "optics"
    AGGLOMERATIVE: Final = "agglomerative"


class MaggScalingMethods:
    STANDARD: Final = "standard"
    MCD: Final = "mcd"
    ROBUST: Final = "robust"


class MaggReductionMethods:
    PCA: Final = "pca"
    IPCA: Final = "ipca"
    KPCA: Final = "kpca"
    FICA: Final = "fica"
    # _LDA : Final = "lda"


# https://scikit-learn.org/stable/modules/clustering.html
_CLUSTERING_ALGOS = {
    MaggClusteringMethods.KMEANS: KMeans,
    MaggClusteringMethods.DBSCAN: DBSCAN,
    MaggClusteringMethods.BIRCH: Birch,
    MaggClusteringMethods.OPTICS: OPTICS,
    MaggClusteringMethods.AGGLOMERATIVE: AgglomerativeClustering,
}

# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
_REDUCTION_ALGOS = {
    MaggReductionMethods.PCA: PCA,
    MaggReductionMethods.IPCA: IncrementalPCA,
    MaggReductionMethods.KPCA: KernelPCA,
    MaggReductionMethods.FICA: FastICA,
}


def _get_clusters(
    data: pd.DataFrame, clust: Union[str, Callable], **kwargs: Dict[Any, Any]
) -> List[int]:
    # map clust to sklearn Class, otherwise default to clust (assuming it's a callable)
    clust_func = _CLUSTERING_ALGOS.get(clust, clust)
    if not callable(clust_func):
        raise InputError(
            f"Invalid cluster function specified; must be one of {list(_CLUSTERING_ALGOS.keys())} or Callable. (Received: {clust})"
        )
    return clust_func(**kwargs).fit_predict(data)


def _scaling(
    data: pd.DataFrame, scale: Union[str, Callable], **kwargs: Dict[Any, Any]
) -> pd.DataFrame:
    if scale == MaggScalingMethods.STANDARD:
        return sklearn_scale(data, **kwargs)
    if scale == MaggScalingMethods.MCD:
        mcd = MinCovDet(**kwargs).fit(data)
        return data - mcd.location_  # centers
    if scale == MaggScalingMethods.ROBUST:
        if not callable(robust_pca):
            raise ImportError(
                "Unable to import `tensorly` library to perform `robust` scaling; run ´pip3 install tensorly` from within your project environment to install it."
            )
        scaled_data, _ = robust_pca(data.values.astype(np.float))
        return scaled_data
    if callable(scale):
        return scale(data, **kwargs)
    raise InputError(
        f"Invalid scaling function specified; must be one of ['standard', 'mcd', 'robust'] or Callable. (Received: {scale})"
    )


def _reduce(
    data: pd.DataFrame, reduct: Union[str, Callable], **kwargs: Dict[Any, Any]
) -> np.ndarray:
    # map reduct to sklearn Decomposition, otherwise default to clust (assuming it's a callable)
    reduct_func = _REDUCTION_ALGOS.get(reduct, reduct)
    if reduct_func in _REDUCTION_ALGOS.values():
        # only handle first component of sklearn.decomposition algorithms
        kwargs["n_components"] = 1
    if not callable(reduct_func):
        raise InputError(
            f"Invalid dimensionality reduction function specified; must be one of {list(_REDUCTION_ALGOS.keys())} or Callable. (Received: {reduct})"
        )
    return reduct_func(**kwargs).fit_transform(data)


@formatting(
    on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=FORMATTING_IGNORE_DTYPES
)  # fmt: off
@validate_numeric(
    on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=VALIDATE_NUMERIC_ON_OUTPUT
)  # fmt: off
@BEARTYPE
def quantile_magg(
    data: pd.DataFrame,
    aggr: int = 2,
    drop_duplicates: bool = True,
    measure: Union[str, Callable] = "mean",
    measure_kwargs: Optional[Dict[Any, Any]] = None,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
) -> pd.DataFrame:
    """Perform microaggregation based on quantiled binning of the data.

    This is a fast and simple (primitive) method of microaggregation, recommended for large datasets.

    Parameters
    ----------
    data : DataFrame
        The data to be modified.
    aggr : int (Default: 2)
        Aggregation level. Must be an integer greater than 0.
    drop_duplicates : bool (Default: True)
        Drop duplicates when creating quantile bins. This may result in a fewer number of
        bins, however, not dropping duplicates will result in a `ValueError` since the
        bin edges will not be unique.
    measure : {"min", "mean", "max", "median"} or Callable (Default: "mean")
        Aggregation statistic. A Callable, such as `scipy.stats.trim_mean` or `np.mean`,
        can be passed for additional functionality. Refer to [1]_ for a complete list of allowed values.
    measure_kwargs : dict
        Keyword arguments to be passed into the `measure` function.
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Returns
    -------
    DataFrame
        A DataFrame with microaggregated values.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.random_integers(0, 100, (10,3)))
       0    1   2
    0  8    57  50
    1  56   44  27
    2  99   25  47
    3  18   21  1
    4  9    28  71
    5  42   69  14
    6  62   24  8
    7  100  44  26
    8  72   98  94
    9  87   40  51

    >>> quantile_magg(df, aggr=2, measure="mean")
       0   1   2
    0  8   57  50
    1  59  42  37
    2  99  26  37
    3  30  22  4
    4  8   26  82
    5  30  83  20
    6  59  22  4
    7  99  42  20
    8  79  83  82
    9  79  42  50

    References
    ----------
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.DataFrameGroupBy.aggregate.html
    """
    if not (0 < aggr <= len(data)):
        raise NotInRangeError(
            f"The aggregation level must be an integer between [1, len(data)={len(data)}]. (Received: {aggr})"
        )

    qs = len(data) // aggr
    qcuts = data.apply(
        lambda col: pd.qcut(
            col,
            q=qs,
            duplicates=MaggConst.DROP if drop_duplicates else MaggConst.RAISE,
        )
    )

    if qcuts.isnull().values.any():
        raise ValueError(
            "Computed quantiles contain NaN(s); verify that `data` is not entirely composed of identical vlaues."
        )

    measure_kwargs = measure_kwargs if measure_kwargs else {}
    gb_vals = data.apply(
        lambda col: col.groupby(qcuts[col.name]).agg(measure, **measure_kwargs)
    )

    data = data.apply(
        lambda col: [gb_vals.loc[i, col.name] for i in qcuts.loc[:, col.name]]
    )

    return data


@formatting(
    on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=FORMATTING_IGNORE_DTYPES
)  # fmt: off
@validate_numeric(
    on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=VALIDATE_NUMERIC_ON_OUTPUT
)  # fmt: off
@BEARTYPE
def sequential_magg(
    data: pd.DataFrame,
    aggr: int = 2,
    sort_all: bool = False,
    sort_by: Optional[str] = None,
    measure: Union[str, Callable] = "mean",
    measure_kwargs: Optional[Dict[Any, Any]] = None,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
) -> Union[pd.DataFrame, pd.Series]:
    """Perform microaggregation based on sequential binning of the data.

    Sorting on a specific variable by specifying `sort_by` is recommended.

    Parameters
    ----------
    data : DataFrame
        The data to be modified.
    aggr : int (Default: 2)
        Aggregation level. Must be an integer greater than 0.
    sort_all : bool (Default: False)
        Aggregate after sorting each column. Cannot be used with `sort_by`.
    sort_by : string, Optional
        Aggregate after sorting by a single column. Cannot be used with `sort_all`.
    measure : {"min", "mean", "max", "median"} or Callable (Default: "mean")
        Aggregation statistic. A Callable, such as `scipy.stats.trim_mean` or `np.mean`,
        can be passed for additional functionality. Refer to [1]_ for a complete list of allowed values.
    measure_kwargs : dict
        Keyword arguments to be passed into the `measure` function.
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Returns
    -------
    DataFrame
        A DataFrame with microaggregated values.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.random_integers(0, 100, (10,3)))
       0    1   2
    0  8    57  50
    1  56   44  27
    2  99   25  47
    3  18   21  1
    4  9    28  71
    5  42   69  14
    6  62   24  8
    7  100  44  26
    8  72   98  94
    9  87   40  51

    >>> sequential_magg(df, aggr=3, measure="min", sort_all=True)
       0    1   2
    0  8    44  50
    1  42   28  26
    2  72   21  26
    3  8    21  1
    4  8    28  50
    5  42   44  1
    6  42   21  1
    7  100  44  26
    8  72   98  94
    9  72   28  50

    References
    ----------
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.DataFrameGroupBy.aggregate.html
    """
    if aggr <= 0:
        raise LessThanOrEqualToZeroError(
            f"The aggregation level must be an integer greater than 0. (Received: {aggr})"
        )

    bins = data.index // aggr

    measure_kwargs = measure_kwargs if measure_kwargs else {}
    if sort_all:

        def _sorted_magg(col):
            svals = col.sort_values()
            gb_svals = svals.groupby(bins).agg(measure, **measure_kwargs)
            svals.iloc[svals.index] = [gb_svals.loc[i] for i in bins]
            return svals.values

        data = data.apply(_sorted_magg)
    elif sort_by:
        svals = data.sort_values(by=sort_by)
        gb_vals = svals.groupby(bins).agg(measure, **measure_kwargs)
        data.iloc[svals.index] = [gb_vals.loc[i] for i in bins]
    else:
        gb_vals = data.groupby(bins).agg(measure, **measure_kwargs)
        data = data.apply(lambda col: [gb_vals.loc[i, col.name] for i in bins])

    return data


@formatting(
    on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=FORMATTING_IGNORE_DTYPES
)  # fmt: off
@validate_numeric(
    on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=VALIDATE_NUMERIC_ON_OUTPUT
)  # fmt: off
@BEARTYPE
def individual_ranking_magg(
    data: pd.DataFrame,
    aggr: int = 2,
    rank_method: str = "first",
    measure: Union[str, Callable] = "mean",
    measure_kwargs: Optional[Dict[Any, Any]] = None,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
) -> Union[pd.DataFrame, pd.Series]:
    """Perform microaggregation based on individual ranking of the data.

    This method preserves univariate statistics best, but significantly alters multivariate statistics.

    Parameters
    ----------
    data : DataFrame
        The data to be modified.
    aggr : int (Default: 2)
        Aggregation level. Must be an integer greater than 0.
    rank_method : {'average', 'min', 'max', 'first', 'dense'}, Optional
        How to rank the group of records that have the same value (i.e. ties):

            * average: average rank of the group
            * min: lowest rank in the group
            * max: highest rank in the group
            * first: ranks assigned in order they appear in the array
            * dense: like 'min', but rank always increases by 1 between groups.

    measure : {"min", "mean", "max", "median"} or Callable (Default: "mean")
        Aggregation statistic. A Callable, such as `scipy.stats.trim_mean` or `np.mean`,
        can be passed for additional functionality. Refer to [1]_ for a complete list of allowed values.
    measure_kwargs : dict
        Keyword arguments to be passed into the `measure` function.
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Returns
    -------
    DataFrame
        A DataFrame with microaggregated values.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.random_integers(0, 100, (10,3)))
       0    1   2
    0  8    57  50
    1  56   44  27
    2  99   25  47
    3  18   21  1
    4  9    28  71
    5  42   69  14
    6  62   24  8
    7  100  44  26
    8  72   98  94
    9  87   40  51

    >>> individual_ranking_magg(df, aggr=2, rank_method="first", measure="mean", keep_dtypes=False)
       0     1     2
    0  8.5   50.5  50.5
    1  59.0  42.0  37.0
    2  99.5  26.5  37.0
    3  30.0  22.5  4.5
    4  8.5   26.5  82.5
    5  30.0  83.5  20.0
    6  59.0  22.5  4.5
    7  99.5  50.5  20.0
    8  79.5  83.5  82.5
    9  79.5  42.0  50.5

    References
    ----------
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.DataFrameGroupBy.aggregate.html
    """
    if aggr <= 0:
        raise LessThanOrEqualToZeroError(
            f"The aggregation level must be an integer greater than 0. (Received: {aggr})"
        )

    svals = data.apply(pd.Series.sort_values, ignore_index=True)
    bins = svals.index // aggr
    ranks = data.rank(method=rank_method).astype(int) - 1  # shift 1 to match index

    measure_kwargs = measure_kwargs if measure_kwargs else {}
    gb_vals = svals.groupby(bins).agg(measure, **measure_kwargs)

    data = data.apply(lambda col: [gb_vals.loc[i, col.name] for i in bins])
    data = data.apply(lambda col: col[ranks[col.name]].values)

    return data


@formatting(
    on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=FORMATTING_IGNORE_DTYPES
)  # fmt: off
@validate_numeric(
    on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=VALIDATE_NUMERIC_ON_OUTPUT
)  # fmt: off
@BEARTYPE
def robust_magg(
    data: pd.DataFrame,
    aggr: int = 2,
    seed: int = 123,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
) -> Union[pd.DataFrame, pd.Series]:
    """Perform microaggregation based on multivariate distances of the data.

    Note that this method uses a Minimum Covariance Determinant (MCD) estimator model that
    is best suited for normally distributed data. Refer to [1]_ for more information.

    Parameters
    ----------
    data : DataFrame or Series
        The data to be modified.
    aggr : int (Default: 2)
        Aggregation level. Must be an integer greater than 0.
    seed : int (Default: 123)
        Random seed to use to fit a Minimum Covariance Determinant (MCD) estimator model.
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Returns
    -------
    DataFrame
        A DataFrame with microaggregated values.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.random_integers(0, 100, (10,3)))
       0    1   2
    0  8    57  50
    1  56   44  27
    2  99   25  47
    3  18   21  1
    4  9    28  71
    5  42   69  14
    6  62   24  8
    7  100  44  26
    8  72   98  94
    9  87   40  51

    >>> robust_magg(df, aggr=3)
       0   1   2
    0  55  65  65
    1  45  29  11
    2  62  41  39
    3  45  29  11
    4  62  41  39
    5  62  41  39
    6  45  29  11
    7  62  41  39
    8  55  65  65
    9  55  65  65

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html
    """
    if aggr <= 0:
        raise LessThanOrEqualToZeroError(
            f"The aggregation level must be an integer greater than 0. (Received: {aggr})"
        )

    if len(data.columns) < 2:
        raise InputError(
            f"Invalid input dimensions. `robust_magg` requires > 1 columns. (Received: {len(data.columns)})"
        )

    # test data for normality; z-scores are only meaningful for normally distributed data
    result = shapiro(data)
    if result.pvalue < 0.05:
        print(
            f"Warning: data not normally distributed; fails Shapiro-Wilk test (p={result.pvalue})."
        )

    z = zscore(data, ddof=1, nan_policy="raise").values

    if np.isnan(z).any():
        raise ValueError(
            "Computed z-scores contain NaN(s); verify that `data` is not entirely composed of identical vlaues."
        )

    pw_dists = pairwise_distances(z)

    if not all(np.diagonal(pw_dists)) == 0:
        np.fill_diagonal(pw_dists, 0)

    mcd = MinCovDet(random_state=seed).fit(z)
    mah_dists = mcd.dist_

    def _knn(pwds, aggr):
        # get k-(min)-nearest neighbors to max (remaining/input) values
        min_val_idxs = [0 for _ in range(aggr)]
        for i in range(aggr):
            min_val_idxs[i] = np.nanargmin(pwds)
            pwds[min_val_idxs[i]] = np.nan
        return min_val_idxs

    for _ in range((len(data) // aggr) - 1):
        max_val_idx = np.nanargmax(mah_dists)
        min_val_idxs = _knn(pw_dists[:, max_val_idx], aggr)
        pw_dists[
            min_val_idxs,
        ] = np.nan
        mah_dists[min_val_idxs] = np.nan
        z[min_val_idxs] = np.mean(z[min_val_idxs], axis=0)

    min_val_idxs = np.unique(
        np.argwhere(~np.isnan(pw_dists))[:, 0]
    )  # get idx of remaining non-nan values
    z[min_val_idxs,] = z[min_val_idxs,].mean(
        axis=0
    )  # merge w above

    mat = (z * data.std().to_numpy()) + data.mean().to_numpy()

    data = pd.DataFrame(mat, columns=data.columns)

    return data


@formatting(
    on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=FORMATTING_IGNORE_DTYPES
)  # fmt: off
@validate_numeric(
    on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=VALIDATE_NUMERIC_ON_OUTPUT
)  # fmt: off
@BEARTYPE
def advanced_magg(
    data: pd.DataFrame,
    aggr: int = 2,
    # clustering method
    clust: Optional[str] = None,
    clust_kwargs: Optional[Dict[Any, Any]] = None,
    # dimensionality reduction method
    reduct: Optional[str] = None,
    reduct_kwargs: Optional[Dict[Any, Any]] = None,
    # data standardizing method;
    scale: Optional[str] = None,
    scale_kwargs: Optional[Dict[Any, Any]] = None,
    # aggregation method
    measure: Union[str, Callable] = "mean",
    measure_kwargs: Optional[Dict[Any, Any]] = None,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
) -> pd.DataFrame:
    """Perform microaggregation based on clustering and/or dimensionality reduction of the data.

    If a clustering method (`clust`) is specified, then the data will be arranged based on the clusters
    they belong to prior to microaggregation. Alternatively, you could specify only a dimensionality reduction method (`reduct`)
    to arrange the data based on their first principle component, or the projection which maximizes the variance in the data.
    Both, clustering and dimensionality reduction methods, can be specified, in which case the data arranged by clusters is used,
    instead of the original data, in dimensionality reduction to re-arrange the data, based on first principal component,
    prior to microaggregation. A scaling method (`scale`) can also be specified to standardize the data prior to performing
    dimensionality reduction. Note that either `clust`, `reduct`, or both must be specified.

    Parameters
    ----------
    data : DataFrame
        The data to be modified.
    aggr : int (Default: 2)
        Aggregation level. Must be an integer greater than 0.
    clust : {"birch, "dbscan", "kmeans", "optics"} or Callable, Optional (Default: None)
        Clustering method to use. The allowed options will use the `sklearn.clustering` methods of the same name.
        Any other `Callable` that adheres to the `sklearn.clustering` API, such as `sklearn.cluster.SpectralClustering`,
        can also be passed for additional functionality. Note that `clust` must be specified if `reduct` is not.
        Refer to [1]_ for more information on the clustering methods being used.
    clust_kwargs : dict, Optional
        Keyword arguments to past into the `clust` function.
        Refer to [1]_ for arguments that can be passed.
    scale : {"standard", "mcd", "robust"} or Callable, Optional (Default: None)
        Scaling method to use. Scaling aims to standarize, or transform, the data by centering to a specfic location (e.g., a cluster or 0).
            - if `scale` is "standard":
                The data will be centered to the mean and scaled to unit variance. Refer to [2]_ for more information.
            - if `scale` is "mcd":
                The data will be centered around the location of the robust MCD estimator. Refer to [3]_ for more information.
            - if `scale` is "robust":
                The data will be centered around decomposition of the input data. Refer to [4]_ for more information.
                Note that this method required the `tensorly` Python library to be installed, which is not installed by default when installing `pymasq`.
        Note that scaling will only occur if `reduct` is passed.
    scale_kwargs : dict, Optional
        Keyword arguments to past into the `scale` function.
        Refer to [2]_, [3]_, and [4]_ for allowed arguments that can be passed to "standard", "mcd", and "robust" methods, repsectively.
    reduct : {"pca", "ipca", "kpca", "fica"} or Callable, Optional (Default: None)
        Dimensionality reduction (`reduct`) method to use. The allowed options are references to the
        `sklearn.decomposiition` methods of dimensionality reduction methods of similar name: `PCA` ("pca"),
        `IncrementalPCA` ("ipca"), KernelPCA ("kpca"), and FastICA ("fica"). Any other `Callabl` that adheres to the
        `sklearn.decopmosition` API, such as `sklearn.decomposition.SparsePCA`, can be passed for additional functionality.
        Note that `reduct` must be specified if `clust` is not.
        Refer to [5]_ for more information on the dimensionality reduction methods being used.
    reduct_kwargs : dict, Optional
        Keyword arguments to pass into the `reduct` function.
        Refer to [5]_ for arguments that can be passed.
    measure : {"min", "mean", "max", "median"} or Callable (Default: "mean")
        Aggregation statistic. A Callable, such as `scipy.stats.trim_mean` or `np.mean`,
        can be passed for additional functionality. Refer to [6]_ for a complete list of allowed values.
    measure_kwargs : dict, Optional
        Keyword arguments to be passed into the `measure` function.
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Returns
    -------
    DataFrame
        A DataFrame with microaggregated values.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(np.random.randint(0, 100, (10,3)))
       0    1   2
    0  8    57  50
    1  56   44  27
    2  99   25  47
    3  18   21  1
    4  9    28  71
    5  42   69  14
    6  62   24  8
    7  100  44  26
    8  72   98  94
    9  87   40  51

    >>> advanced_magg(df, aggr=3, clust="kmeans")
       0    1   2
    0  64   40  49
    1  63   55  43
    2  64   40  49
    3  23   39  28
    4  23   39  28
    5  23   39  28
    6  63   55  43
    7  100  44  26
    8  63   55  43
    9  64   40  49

    >>> advanced_magg(df, aggr=3, scale="standard", reduct="pca")
       0   1   2
    0  34  41  57
    1  45  29  12
    2  80  46  29
    3  45  29  12
    4  34  41  57
    5  80  46  29
    6  45  29  12
    7  80  46  29
    8  72  98  94
    9  34  41  57

    >>> advanced_magg(df, aggr=3, clust="kmeans", scale="standard", reduct="pca")
       0    1   2
    0  64   40  49
    1  48   54  40
    2  64   40  49
    3  48   54  40
    4  37   40  31
    5  37   40  31
    6  37   40  31
    7  100  44  26
    8  48   54  40
    9  64   40  49

    >>> advanced_magg(df, aggr=3, clust="kmeans", clust_kwargs={"n_clusters": 3},
                      scale="robust", scale_kwargs={"learning_rate": 0.05},
                      reduct="fica", reduct_kwargs={"max_iter": 300})
       0   1   2
    0  65  47  42
    1  72  31  27
    2  72  31  27
    3  23  39  28
    4  23  39  28
    5  23  39  28
    6  72  31  27
    7  65  47  42
    8  72  98  94
    9  65  47  42

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/clustering.html
    [2] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html
    [3] https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html
    [4] http://tensorly.org/stable/modules/generated/tensorly.decomposition.robust_pca.html
    [5] https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
    [6] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.DataFrameGroupBy.aggregate.html
    """
    if not clust and not reduct:
        raise InputError(
            "Must specify clustering and/or dimensionality reduction method, `clust` and/or `reduct`, respectively."
        )

    if aggr <= 0:
        raise LessThanOrEqualToZeroError(
            f"The aggregation level must be an integer greater than 0. (Received: {aggr})"
        )

    bins = data.index // aggr

    c_idx, _data = None, None
    if clust:
        # cluster similar records
        clust_kwargs = clust_kwargs if clust_kwargs else {}
        clusters = _get_clusters(data, clust, **clust_kwargs)
        c_idx = np.argsort(clusters)
        _data = data.iloc[c_idx]

    r_idx = None
    if reduct:
        # perform dimensionality reduction on the data or clustered data
        scaled_data = None
        if scale:
            # preprocess the data by scaling, or standardizing, it prior to dimensionality reduction
            scale_kwargs = scale_kwargs if scale_kwargs else {}
            scaled_data = _scaling(
                _data if _data is not None else data, scale, **scale_kwargs
            )
        reduct_kwargs = reduct_kwargs if reduct_kwargs else {}
        pc = _reduce(scaled_data if scale else data, reduct, **reduct_kwargs)
        r_idx = np.argsort(np.ravel(pc))

    measure_kwargs = measure_kwargs if measure_kwargs else {}

    # aggregate the data
    if clust and reduct is None:
        # aggregate based on clusters only; no dimensionality reduction applied
        gb_vals = data.iloc[c_idx].groupby(bins).agg(measure, **measure_kwargs)
        data.iloc[c_idx] = [gb_vals.loc[i] for i in bins]
    elif clust is None and reduct:
        # no clustering applied; aggregate based on dimensionality reduction only
        gb_vals = data.iloc[r_idx].groupby(bins).agg(measure, **measure_kwargs)
        data.iloc[r_idx] = [gb_vals.loc[i] for i in bins]
    else:
        # aggregate based on applied clustering and dimensionality reduction
        bins = data.iloc[r_idx].index // aggr
        gb_vals = _data.iloc[r_idx].groupby(bins).agg(measure, **measure_kwargs)
        _data.iloc[r_idx] = [gb_vals.loc[i] for i in bins]
        data = _data.sort_index()

    return data


def microaggregation(
    data: Union[pd.DataFrame, pd.Series],
    method: str,
    **kwargs,
) -> pd.DataFrame:
    """Perform data microaggregation.

    Wrapper function for `quantile_magg`, `sequential_magg`, `ranking_magg`, and `robust_magg`.

    Records are grouped based on a proximity measure of variables of interest, and the same small
    groups of records are used in calculating aggregates for those variables. The aggregates are
    released instead of the individual record values.

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The data to be modified.
    method : {'quantile', 'sequential', 'ranking', 'robust'}
        The microaggregation method to perform.
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    **kwargs
        Additional keyword arguments to be passed to `quantile_magg`, `sequential_magg`, `ranking_magg`, `robust_magg`, or `advanced_magg`.

        * aggr : int (Default: 2)
            Aggregation level.

        * keep_dtypes : bool (Default: True)
            Determine whether the data types of the output values should be the same as the
            data types of the input values.

        if `method` is `quantile`:
            * drop_duplicates : bool (Default: True)
                Drop duplicates when creating quantile bins. This may result in a fewer number of
                bins, however, not dropping duplicates will result in a `ValueError` since the
                bin edges will not be unique.
            * measure : {"min", "mean", "max", "median"} or Callable (Default: "mean")
                Aggregation statistic. A Callable, such as `scipy.stats.trim_mean` or `np.mean`,
                can be passed for additional functionality.
            * measure_kwargs : dict
                Keyword arguments to be passed into the `measure` function.

        if `method` is `sequential`:
            * sort_all : bool (Default: False)
                Aggregate after sorting each column. Cannot be used with `sort_by`.
            * sort_by : string, Optional
                Aggregate after sorting by a single column. Cannot be used with `sort_all`.
            * measure : {"min", "mean", "max", "median"} or Callable (Default: "mean")
                Aggregation statistic. A Callable, such as `scipy.stats.trim_mean` or `np.mean`,
                can be passed for additional functionality.
            * measure_kwargs : dict
                Keyword arguments to be passed into the `measure` function.

        if "method" is "ranking":
            * rank_method : {'average', 'min', 'max', 'first', 'dense'}, Optional
                How to rank the group of records that have the same value (i.e. ties):

                    * average: average rank of the group
                    * min: lowest rank in the group
                    * max: highest rank in the group
                    * first: ranks assigned in order they appear in the array
                    * dense: like 'min', but rank always increases by 1 between groups.

            * measure : {"min", "mean", "max", "median"} or Callable (Default: "mean")
                Aggregation statistic. A Callable, such as `scipy.stats.trim_mean` or `np.mean`,
                can be passed for additional functionality.
            * measure_kwargs : dict
                Keyword arguments to be passed into the `measure` function.

        if `method` is "robust":
            * seed : int (Default: 1)
                Random seed to use to fit a Minimum Covariance Determinant (MCD) estimator model.

        if `method` is "advanced":
            * clust : {"birch, "dbscan", "kmeans", "optics"} or Callable, Optional (Default: None)
                Clustering method to use. The allowed options will use the `sklearn.clustering` methods of the same name.
                Any other `Callable` that adheres to the `sklearn.clustering` API, such as `sklearn.cluster.SpectralClustering`,
                can also be passed for additional functionality. Note that `clust` must be specified if `reduct` is not.
                Refer to [1]_ for more information on the clustering methods being used.
            * clust_kwargs : dict, Optional
                Keyword arguments to past into the `clust` function.
                Refer to [1]_ for arguments that can be passed.
            * scale : {"standard", "mcd", "robust"} or Callable, Optional (Default: None)
                Scaling method to use. Scaling aims to standarize, or transform, the data by centering to a specfic location (e.g., a cluster or 0).
                    - if `scale` is "standard":
                        The data will be centered to the mean and scaled to unit variance. Refer to [2]_ for more information.
                    - if `scale` is "mcd":
                        The data will be centered around the location of the robust MCD estimator. Refer to [3]_ for more information.
                    - if `scale` is "robust":
                        The data will be centered around decomposition of the input data. Refer to [4]_ for more information.
                        Note that this method required the `tensorly` Python library to be installed, which is not installed by default when installing `pymasq`.
                Note that scaling will only occur if `reduct` is passed.
            * scale_kwargs : dict, Optional
                Keyword arguments to past into the `scale` function.
                Refer to [2]_, [3]_, and [4]_ for allowed arguments that can be passed to "standard", "mcd", and "robust" methods, repsectively.
            * reduct : {"pca", "ipca", "kpca", "fica"} or Callable, Optional (Default: None)
                Dimensionality reduction (`reduct`) method to use. The allowed options are references to the
                `sklearn.decomposiition` methods of dimensionality reduction methods of similar name: `PCA` ("pca"),
                `IncrementalPCA` ("ipca"), KernelPCA ("kpca"), and FastICA ("fica"). Any other `Callabl` that adheres to the
                `sklearn.decopmosition` API, such as `sklearn.decomposition.SparsePCA`, can be passed for additional functionality.
                Note that `reduct` must be specified if `clust` is not.
                Refer to [5]_ for more information on the dimensionality reduction methods being used.
            * reduct_kwargs : dict, Optional
                Keyword arguments to pass into the `reduct` function.
                Refer to [5]_ for arguments that can be passed.
            * measure : {"min", "mean", "max", "median"} or Callable (Default: "mean")
                Aggregation statistic. A Callable, such as `scipy.stats.trim_mean` or `np.mean`,
                can be passed for additional functionality. Refer to [6]_ for a complete list of allowed values.
            * measure_kwargs : dict, Optional
                Keyword arguments to be passed into the `measure` function.

    Returns
    -------
    DataFrame
        A DataFrame with microaggregated values.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.random_integers(0, 100, (10,3)))
       0    1   2
    0  8    57  50
    1  56   44  27
    2  99   25  47
    3  18   21  1
    4  9    28  71
    5  42   69  14
    6  62   24  8
    7  100  44  26
    8  72   98  94
    9  87   40  51

    >>> microaggregation(df, method="quantile", aggr=2, measure="mean")
       0   1   2
    0  8   57  50
    1  59  42  37
    2  99  26  37
    3  30  22  4
    4  8   26  82
    5  30  83  20
    6  59  22  4
    7  99  42  20
    8  79  83  82
    9  79  42  50

    >>> microaggregation(df, method="sequential", aggr=3, measure="min", sort_all=True)
       0    1   2
    0  8    44  50
    1  42   28  26
    2  72   21  26
    3  8    21  1
    4  8    28  50
    5  42   44  1
    6  42   21  1
    7  100  44  26
    8  72   98  94
    9  72   28  50

    >>> microaggregation(df, method="ranking", aggr=2, rank_method="first", measure="mean", keep_dtypes=False)
       0     1     2
    0  8.5   50.5  50.5
    1  59.0  42.0  37.0
    2  99.5  26.5  37.0
    3  30.0  22.5  4.5
    4  8.5   26.5  82.5
    5  30.0  83.5  20.0
    6  59.0  22.5  4.5
    7  99.5  50.5  20.0
    8  79.5  83.5  82.5
    9  79.5  42.0  50.5

    >>> microaggregation(df, method="robust", aggr=3)
       0   1   2
    0  55  65  65
    1  45  29  11
    2  62  41  39
    3  45  29  11
    4  62  41  39
    5  62  41  39
    6  45  29  11
    7  62  41  39
    8  55  65  65
    9  55  65  65

    >>> microaggregation(df, method="advanced", aggr=3, clust="optics")
       0   1    2
    0  54  42  41
    1  54  42  41
    2  54  42  41
    3  23  39  28
    4  23  39  28
    5  23  39  28
    6  78  55  42
    7  78  55  42
    8  78  55  42
    9  87  40  51

    See Also
    --------
    pymasq.mitigations.quantile_magg : Perform microaggregation based on quantiled binning of the data.

    pymasq.mitigations.sequential_magg : Perform microaggregation based on sequential binning of the data.

    pymasq.mitigations.ranking_magg : Perform microaggregation based on individual ranking of the data.

    pymasq.mitigations.robust_magg : Perform microaggregation based on multivariate distances of the data.

    pymasq.mitigations.advanced_magg : Perform microaggregation based on clustering and/or dimensionality reduction of the data.
    """
    if method == MaggMethods.QUANTILE:
        return quantile_magg(
            data,
            aggr=kwargs.get("aggr", 2),
            drop_duplicates=kwargs.get("drop_duplicates", True),
            measure=kwargs.get("measure", "mean"),
            measure_kwargs=kwargs.get("measure_kwargs", None),
            cols=kwargs.get("cols", None),
            keep_dtypes=kwargs.get("keep_dtypes", True),
        )
    elif method == MaggMethods.SEQUENTIAL:
        return sequential_magg(
            data,
            aggr=kwargs.get("aggr", 2),
            sort_all=kwargs.get("sort_all", False),
            sort_by=kwargs.get("sort_by", None),
            measure=kwargs.get("measure", "mean"),
            measure_kwargs=kwargs.get("measure_kwargs", None),
            cols=kwargs.get("cols", None),
            keep_dtypes=kwargs.get("keep_dtypes", True),
        )
    elif method == MaggMethods.RANKING:
        return individual_ranking_magg(
            data,
            aggr=kwargs.get("aggr", 2),
            rank_method=kwargs.get("rank_method", "first"),
            measure=kwargs.get("measure", "mean"),
            measure_kwargs=kwargs.get("measure_kwargs", None),
            cols=kwargs.get("cols", None),
            keep_dtypes=kwargs.get("keep_dtypes", True),
        )
    elif method == MaggMethods.ROBUST:
        return robust_magg(
            data,
            aggr=kwargs.get("aggr", 2),
            seed=kwargs.get("seed", 123),
            cols=kwargs.get("cols", None),
            keep_dtypes=kwargs.get("keep_dtypes", True),
        )
    elif method == MaggMethods.ADVANCED:
        return advanced_magg(
            data,
            aggr=kwargs.get("aggr", 2),
            clust=kwargs.get("clust", None),
            clust_kwargs=kwargs.get("clust_kwargs", None),
            reduct=kwargs.get("reduct", None),
            reduct_kwargs=kwargs.get("reduct_kwargs", None),
            scale=kwargs.get("scale", None),
            scale_kwargs=kwargs.get("scale_kwargs", None),
            measure=kwargs.get("measure", "mean"),
            measure_kwargs=kwargs.get("measure_kwargs", None),
            cols=kwargs.get("cols", None),
            keep_dtypes=kwargs.get("keep_dtypes", True),
        )

    raise InputError(
        f"Invalid `method` defined; method must be one of ['{MaggMethods.QUANTILE}', '{MaggMethods.RANKING}', '{MaggMethods.SEQUENTIAL}', '{MaggMethods.ROBUST}', '{MaggMethods.ADVANCED}']. (Received: {method})"
    )
