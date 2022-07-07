import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from scipy.stats import chi2
from scipy.stats import multivariate_normal
from sklearn.preprocessing import LabelEncoder
from sklearn.covariance import MinCovDet
from typing import List, Optional, Union, Final

from pymasq.config import (
    FORMATTING_ON_OUTPUT,
    VALIDATE_NUMERIC_ON_INPUT,
    VALIDATE_NUMERIC_ON_OUTPUT,
)
from pymasq.mitigations.utils import _as_series, _as_dataframe
from pymasq.utils import validate_numeric, formatting
from pymasq import BEARTYPE
from pymasq.errors import InputError

__all__ = [
    "add_noise",
    "add_noise_additive",
    "add_noise_correlated",
    "add_noise_restricted",
    "add_noise_outliers",
    "ADDITIVE",
    "CORRELATED",
    "RESTRICTED",
    "OUTLIERS",
]


ADDITIVE: Final = "additive"
CORRELATED: Final = "correlated"
RESTRICTED: Final = "restricted"
OUTLIERS: Final = "outliers"


class OUTLIERS_INTERPOLATION_METHODS:
    LINEAR = "linear"
    LOWER = "lower"
    HIGHER = "higher"
    MIDPOINT = "midpoint"
    NEAREST = "nearest"


@formatting(on_output=FORMATTING_ON_OUTPUT)
@validate_numeric(
    on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=VALIDATE_NUMERIC_ON_OUTPUT
)  # fmt: off
@BEARTYPE
def add_noise_additive(
    data: Union[pd.DataFrame, pd.Series],
    magnitude: Union[int, float] = 5,
    centered: bool = False,
    degrees: int = 1,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
) -> pd.DataFrame:
    """Add Gaussian noise to data.

    Parameters
    ----------
    data : DataFrame or Series
        The data to be modified.
    magnitude : int (Default: 5)
        Amount of perturbative noise (in percentages) to add.
    centered : bool (Default: False)
        Set the mean ("center") of the distribution. If False, the added noise will be
        centered around 0.0, otherwise it will be centered around the mean, relative to the `magnitude`.
    degrees : int (Default: 1)
        Number of degrees of freedom. Must be greater than or equal to 0.
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Returns
    -------
    DataFrame
        A DataFrame perturbed with additive noise.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.random_integers(0, 100, (10,3)))
        0  1  2
    0  182  529  949
    1  783  375  59
    2  253  349  631
    3  889  683  996
    4  523  554  803
    5  279  379  242
    6  639  26   833
    7  30   307  551
    8  681  885  822
    9  659  180  823

    >>> add_noise_additive(df, magnitude=2, centered=True)
        0  1  2
    0  188  526  948
    1  779  383  63
    2  252  346  646
    3  891  690  992
    4  520  555  800
    5  274  380  243
    6  642  18   838
    7  30   299  552
    8  682  890  814
    9  656  186  821
    """
    std = data.std(ddof=degrees)
    magnitude /= 100.0
    if centered:
        delta = np.sqrt(1 - np.square(magnitude))
        loc = (1 - delta) / magnitude
        noise = np.random.normal(loc=loc * data.mean(), scale=std, size=data.shape)
        data *= delta
        return data.add(magnitude * noise)
    return data + np.random.normal(scale=magnitude * std, size=data.shape)


@formatting(on_output=FORMATTING_ON_OUTPUT)
@validate_numeric(
    on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=VALIDATE_NUMERIC_ON_OUTPUT
)  # fmt: off
@BEARTYPE
def add_noise_correlated(
    data: pd.DataFrame,
    magnitude: Union[int, float] = 5,
    encode_non_numeric: bool = False,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
) -> pd.DataFrame:
    """Add correlated noise to data.

      Note that this method requires 2 or more data features, or columns.

      Parameters
      ----------
      data : DataFrame
          The data to be modified.
      magnitude : int (Default: 5)
          Amount of perturbative noise to add. This will be converted into a percentage
          (e.g., magnitude=5 will add 5% noise).
      encode_non_numeric : bool (Default: False)
          If True, all non-numeric columns in `data` will be encoded as numerical values
          using a LabelEncoder and be included in the covariance matrix calculations (note
          that this will result in non-numeric columns being returned as numeric features).
          If False, only numeric columns will be accounted for.
      cols : str or list
          The name of the column or columns to subset from `data` if `data` is a dataframe.
      keep_dtypes : bool (Default: True)
          Determine whether the data types of the output values should be the same as the
          data types of the input values.

      Returns
      -------
      DataFrame
          A DataFrame perturbed with correlated noise.

      Raises
      ------
      InputError:
          This error is raised when an insufficient number of columns are specified.

      Examples
      --------
      >>> df = pd.DataFrame(np.random.random_integers(0, 100, (10,3)))
         0    1    2
      0  182  529  949
      1  783  375  59
      2  253  349  631
      3  889  683  996
      4  523  554  803
      5  279  379  242
      6  639  26   833
      7  30   307  551
      8  681  885  822
      9  659  180  823

      >>> add_noise_correlated(df, magnitude=2, keep_dtypes=True)
          0   1    2
      0  146  525  944
      1  772  351  65
      2  286  344  601
      3  903  664  947
      4  520  543  827
      5  296  375  257
      6  588  30   821
      7  32   346  581
      8  630  893  787
      9  639  158  857

      >>> df2 = pd.DataFrame(np.random.random_integers(1, 100, (10, 3)))
      >>> df2[3] = np.random.choice(["a", "b", "c"], size=10)
          0  1   2   3
      0  98  3   39  b
      1  37  8   56  a
      2  31  68  83  c
      3  18  71  64  b
      4  33  63  3   c
      5  38  20  20  c
      6  23  33  52  a
      7  52  32  15  a
      8  28  95  65  b
      9  58  80  73  a

    >>> add_noise_correlated(df, magnitude=2, encode_non_numeric=True, keep_dtypes=True)
          0  1   2   3
      0  98  2   38  0.986802
      1  37  7   55  0.00440759
      2  31  67  83  1.98077
      3  19  70  63  1.00325
      4  32  63  2   1.98965
      5  37  19  20  1.98499
      6  23  32  52  -0.0159703
      7  51  30  14  0.012063
      8  28  94  65  1.0021
      9  57  80  73  0.0147035
    """
    if len(data.columns) <= 1:
        raise InputError(
            "Correlated noise can only be applied to input `data` with 2+ columns."
        )

    if encode_non_numeric:
        magnitude = np.square(magnitude) / 100.0
        data_encoded = pd.DataFrame(
            [
                data[column[0]]  # takes the column as is if it is numeric
                if is_numeric_dtype(column[1])
                else pd.Series(
                    LabelEncoder().fit_transform(data[column[0]]),
                    name=column[0]  # Label encodes
                    # non-numeric data, preserving the original column name
                )
                for column in data.dtypes.iteritems()
            ]
        ).transpose()  # Transposes the data to have the column/row orientation match the input data

        return data_encoded + np.random.multivariate_normal(
            pd.Series([0] * data_encoded.shape[1]),
            (magnitude / 100.0) * data_encoded.cov(),
            size=data_encoded.shape[0],
        )

    cov_mat = (magnitude / 100.0) * np.cov(data, rowvar=False)
    return data + multivariate_normal.rvs(cov=cov_mat, size=data.shape[0])


@formatting(on_output=FORMATTING_ON_OUTPUT)
@validate_numeric(
    on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=VALIDATE_NUMERIC_ON_OUTPUT
)  # fmt: off
@BEARTYPE
def add_noise_restricted(
    data: pd.DataFrame,
    magnitude: Union[int, float] = 5,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
) -> pd.DataFrame:
    """Add noise relative to the sample size of the data.

    Parameters
    ----------
    data : DataFrame
        The data to be modified.
    magnitude : int (Default: 5)
        Amount of perturbative noise to add. This will be converted into a percentage
        (e.g., magnitude=5 will add 5% noise).
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Returns
    -------
    DataFrame
        A DataFrame with noise added to it.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.random_integers(0, 100, (10,3)))
        0   1    2
    0  182  529  949
    1  783  375  59
    2  253  349  631
    3  889  683  996
    4  523  554  803
    5  279  379  242
    6  639  26   833
    7  30   307  551
    8  681  885  822
    9  659  180  823

    >>> add_noise_restricted(df, magnitude = 2, keep_dtypes=True)
        0   1    2
    0  214  518  919
    1  752  380  123
    2  278  357  635
    3  847  655  961
    4  519  540  789
    5  301  384  287
    6  623  68   815
    7  78   319  563
    8  661  836  806
    9  641  205  806
    """
    nrows = len(data)

    magnitude /= 100.0
    if magnitude > nrows - 1:
        raise InputError(
            f"The value of `magnitude` must be less than the number of rows. (Received: {magnitude*100.0})"
        )

    if nrows < 500:
        cc = np.sqrt((nrows - 1 - magnitude) / ((nrows + 1) * (1 + magnitude)))
    else:
        cc = np.sqrt((nrows - 1) / (nrows + nrows * magnitude - 1))
    delta = (1 - cc) * data.mean()
    return (cc * data) + delta.values


@formatting(on_output=FORMATTING_ON_OUTPUT)
@validate_numeric(
    on_input=VALIDATE_NUMERIC_ON_INPUT, on_output=VALIDATE_NUMERIC_ON_OUTPUT
)  # fmt: off
@BEARTYPE
def add_noise_outliers(
    data: pd.DataFrame,
    magnitude: Union[int, float] = 5,
    q: float = 0.99,
    interpolation: str = "linear",
    seed: int = 123,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
) -> pd.DataFrame:
    """Add noise to outliers of the data.

    Note that this method requires 2 or more data features, or columns.

    Parameters
    ----------
    data : DataFrame
        The data to be modified.
    magnitude : int (Default: 5)
        Amount of perturbative noise to add. This will be converted into a percentage
        (e.g., magnitude=5 will add 5% noise).
    q : float
        The quantile(s) to compute. Must be between 0 <= q <= 1.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'} (Default: 'linear')
        The interpolation method to use. Refer to [1]_ for a complete list of allowed values.
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
        A DataFrame with noise added to its outliers.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.random_integers(0, 100, (10,3)))
        0   1    2
    0  182  529  949
    1  783  375  59
    2  253  349  631
    3  889  683  996
    4  523  554  803
    5  279  379  242
    6  639  26   833
    7  30   307  551
    8  681  885  822
    9  659  180  823

    >>> add_noise_outliers(df, magnitude = 2, q = 0.99, seed=123)
        0   1    2
    0  182  526  944
    1  785  378  63
    2  253  349  631
    3  895  685  995
    4  523  554  803
    5  283  382  243
    6  639  26   833
    7  30   307  551
    8  685  880  824
    9  659  180  823

    References
    ----------
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html
    """
    if len(data.columns) <= 1:
        raise InputError(
            "Correlated noise can only be applied to input `data` with 2+ columns."
        )

    quants = data.quantile(q=q, interpolation=interpolation)
    quant_outliers, _ = np.where(data > quants)

    mcd = MinCovDet(random_state=seed).fit(data)
    dists = np.sqrt(mcd.dist_)

    limit = np.sqrt(chi2.ppf(0.975, len(data.columns)))
    dist_outliers = np.where(dists > limit)[0]

    outliers = np.unique(np.append(quant_outliers, dist_outliers))

    std = 1.96 * data.std() / np.sqrt(len(data)) * (magnitude / 100.0)
    noise = np.random.normal(scale=std, size=(len(outliers), len(data.columns)))

    data.iloc[outliers, :] += noise

    return data


def add_noise(
    data: Union[pd.DataFrame, pd.Series],
    method: str,
    **kwargs,
) -> Union[pd.DataFrame, pd.Series]:
    """Perturb numerical data with noise.

    Wrapper function for `add_noise_additive`, `add_noise_correlated`,
    `add_noise_restricted`, and `add_noise_outliers`.

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The data to be modified. A `DataFrame` is required`add_noise` if `method` is "correlated".
    method : {'additive', 'correlated', 'restricted', 'outliers'}, optional (Default: 'additive')
        The perturbative noise method.
    cols : str or list, Optional
        The name of the column or columns to subset from `data` if `data` is a dataframe. For
        additive noise, a column is required to be provided.
    magnitude : int or float, optional
        The amount of noise (in percentages) to add (Default: 5)
    **kwargs
        Additional argument(s) to be passed to `add_noise_additive`.

        if `method` is 'additive':

            * degrees : int, optional (Default: 1)
                The degrees of freedom for calculating standard deviation.
            * centered : bool (Default: False)
                Set the mean (“center”) of the distribution.

        if `method` is `correlated`:
            * encode_non_numeric : bool (Default: False)
                Encode non-numeric columns as numerical vectors to perform the covariance matrix calculations.
                If False, only numeric columns will be accounted for.

        if `method` is `restricted`:
            _None_

        if `method` is `outliers`:
            * q : float (Default: 0.99)
            * interpolation : str (Default: "linear")
            * seed: int (Default: 123)

    Returns
    -------
    DataFrame
        A DataFrame with noise added to it.

    Raises
    ------
    InputError:
        This error is raised when invalid arguments are passed.

    DataTypeError:
        This error is raised when the parameters supplied for a given method do not match the
        appropriate type of the expected parameters of that method.

    See Also
    --------
    pymasq.mitigations.add_noise_additive : Add Gaussian noise to data.

    pymasq.mitigations.add_noise_correlated : Add correlated noise to data.

    pymasq.mitigations.add_noise_restricted : Add noise relative to the sample size of the data.

    pymasq.mitigations.add_noise_outliers : Add noise to outliers of the data.
    """
    if method == ADDITIVE:
        return add_noise_additive(
            data,
            magnitude=kwargs.get("magnitude", 5),
            centered=kwargs.get("centered", False),
            degrees=kwargs.get("degrees", 1),
            cols=kwargs.get("cols", None),
            keep_dtypes=kwargs.get("keep_dtypes", True),
        )
    elif method == CORRELATED:
        return add_noise_correlated(
            data,
            magnitude=kwargs.get("magnitude", 5),
            encode_non_numeric=kwargs.get("encode_non_numeric", False),
            cols=kwargs.get("cols", None),
            keep_dtypes=kwargs.get("keep_dtypes", True),
        )
    elif method == RESTRICTED:
        return add_noise_restricted(
            data,
            magnitude=kwargs.get("magnitude", 5),
            cols=kwargs.get("cols", None),
            keep_dtypes=kwargs.get("keep_dtypes", True),
        )
    elif method == OUTLIERS:
        return add_noise_outliers(
            data,
            magnitude=kwargs.get("magnitude", 5),
            q=kwargs.get("q", 0.99),
            interpolation=kwargs.get("interpolation", "linear"),
            seed=kwargs.get("seed", 123),
            cols=kwargs.get("cols", None),
            keep_dtypes=kwargs.get("keep_dtypes", True),
        )

    raise InputError(
        f"Invalid `method` defined; method must be one of ['additive', 'correlated', 'restricted', 'outliers']. (Received: {method})"
    )
