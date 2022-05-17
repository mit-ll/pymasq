import itertools

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats
from typing import List, Optional, Union

from pymasq import BEARTYPE
from pymasq.config import FORMATTING_ON_OUTPUT, FORMATTING_IGNORE_DTYPES
from pymasq.errors import InputError
from pymasq.mitigations.utils import _is_identical
from pymasq.utils import formatting, validate_numeric


__all__ = ["geom_transform"]


SKIP_ROTATION_ANGLES = [30, 45, 60, 90, 120, 135, 150, 180]
MAX_DEGREES = 180


@BEARTYPE
def _get_coordinates(ncols: int):
    # get combination of sin/cosine indices
    coords = []
    combs = np.array(list(itertools.combinations(range(ncols), 2)))
    for comb in combs:
        coords.append(np.array(list(itertools.product(comb, comb))))
    return coords


@BEARTYPE
def _get_rotation_plane(ncols: int, coords: list, rad_value: float):
    rot_plane = np.eye(ncols)
    for coord in coords:
        rot_plane_tmp = np.eye(ncols)
        rot_plane_tmp[coord[0, 0], coord[0, 1]] = np.cos(rad_value)
        rot_plane_tmp[coord[1, 0], coord[1, 1]] = -np.sin(rad_value)
        rot_plane_tmp[coord[2, 0], coord[2, 1]] = np.sin(rad_value)
        rot_plane_tmp[coord[3, 0], coord[3, 1]] = np.cos(rad_value)

        rot_plane = rot_plane @ rot_plane_tmp

    return rot_plane


@BEARTYPE
def _gen_rotation_parameters(ncols: int, coords: List, cov: np.ndarray):
    # Identifying the best perturbation parameters for data based on covariance
    rot_params = np.zeros((MAX_DEGREES, ncols))

    for col in range(ncols):
        id_ref = np.eye(ncols)  # identity matrix for reference
        id_ref[:, col] = -id_ref[:, col]

        for rot_angle in range(1, MAX_DEGREES + 1):
            if rot_angle in SKIP_ROTATION_ANGLES:
                continue

            # Generate the rotational matrix
            rad_value = rot_angle * np.pi / 180.0
            rot_plane = _get_rotation_plane(ncols, coords, rad_value)

            rot_diag = np.diag(rot_plane @ (id_ref @ cov @ id_ref.T) @ rot_plane.T)
            rot_cov_sum = np.sum(((cov @ id_ref) * rot_plane).T, axis=0).T

            # subplane rotational matrix of the desired rotation angle
            rot_mat = (1 + rot_diag) - (2 * rot_cov_sum)

            min_val = min(rot_mat)

            rot_params[rot_angle - 1, col] = min_val

    return rot_params


@formatting(
    on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=FORMATTING_IGNORE_DTYPES
)  # fmt: off
@BEARTYPE
def geom_transform(
    data: pd.DataFrame,
    perturb_cols: Optional[List] = None,
    sensitive_col: Optional[Union[str, int]] = None,
    magnitude: Union[int, float] = 0.3,
    shuffle: bool = True,
    cols: Optional[Union[List, str, int]] = None,
    keep_dtypes: bool = True,
):
    """Perform geometric transformations on data via PABIDOT algorithm.

    The Privacy preservation Algorithm for Big Data Using Optimal geometric Transformations (PABIDOT) algorithm
    is an irreversible input perturbation mechanism with a new privacy model (Φ − separation) which facilitates full data release.
    Φ − separation provides an empirical privacy guarantee against data reconstruction attacks.
    PABIDOT is substantially faster than comparable methods; it sequentially applies random axis reflection,
    noise translation, and multidimensional concatenated subplane rotation followed by randomized expansion and random tuple shuffling
    for further randomization. Randomized expansion is a novel method to increase the positiveness or the
    negativeness of a particular data instance. PABIDOT’s memory overhead is comparatively close to other solutions,
    and it provides better attack resistance, classification accuracy, and excellent efficiency towards big data. [1]

    Code was adapted from the author's official github repo [2].

    Parameters
    ----------
    data : DataFrame
        The data to be modified.
    perturb_cols : List of str or ints, , Optional (Default: None)
        The subset of columns that will be manipulated from `data`. The values of `data[perturb_cols]`
        must be numeric and there can be no overlapping columns with `sensitive_col`. If omitted,
        all columns of `data` will be perturbed.
    sensitive_col : str or int, Optional (Default: None)
        The subset of columns that should not be perturbed and instead used as a reference when shuffling records (rows).
        If defined, then only the columns specified in `perturb_cols` will be modified.
        There can be no overlapping columns with `perturb_cols`.
    magnitude : int or float (Default: 0.3)
        Amount of perturbative noise (in percentages) to add.
    shuffle : bool (Default: True)
        Perform random record shuffling for further randomization. If set to True, then
        records (rows) will be shuffled at random. Columns not specified in `perturb_cols` or
        `sensitve_cols` will not be shuffled.
    cols : str or list
        The name of the column or columns to subset from `data` if `data` is a dataframe.
    keep_dtypes : bool (Default: True)
        Determine whether the data types of the output values should be the same as the
        data types of the input values.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.random_integers(0, 100, (10,3)))
        0   1   2   3
    0  72  13  92  91
    1  55  63  65  76
    2  36  61  10  63
    3  60  55  58  22
    4  33   7  42  73
    5  70  62  70  71
    6  59  71  61  92
    7  16   2  87  36
    8   3  40  83  40
    9  98  19   1  81

    >>> geom_transform(df, perturb_cols=[0,1])
               0          1   2   3
    0  71.648190  51.070171  42  73
    1  -9.134024  23.795086  92  91
    2   6.611443  97.804900  87  36
    3  -2.610105  69.084472  65  76
    4  23.181531  -6.396323  61  92
    5 -19.614609   4.535011  10  63
    6  43.362837  55.888641  83  40
    7  58.018783  61.896669   1  81
    8  60.973738  49.287759  58  22
    9  62.473920  20.604887  70  71

    >>> geom_transform(df, perturb_cols=[0,1], sensitive_col=3)
               0           1   2   3
    0 -11.316355   50.629356  92  73
    1  28.481677   17.091661  65  40
    2  -4.765645   93.733731  10  91
    3  72.519084   80.838710  58  92
    4  64.041619   89.521283  42  71
    5  56.931659   55.653460  70  63
    6   2.980840  119.995536  61  81
    7 -15.246824   28.281672  87  36
    8  53.494104   82.965545  83  22
    9  67.496956   70.102160   1  76

    >>> geom_transform(df, perturb_cols=[0,1], sensitive_col=3, shuffle=False)
               0           1   2   3
    0  16.245920   82.742434  92  91
    1  76.375344   68.423197  65  76
    2  74.066167   50.099138  10  63
    3  66.465686   67.654576  58  22
    4   6.436980   49.678812  42  73
    5  74.770801   77.528999  70  71
    6  85.229085   69.590935  61  92
    7   1.085223   20.477899  87  36
    8  41.831609    6.293705  83  40
    9  24.960647  108.288109   1  81

    >>> geom_transform(df, perturb_cols=[0,1], sensitive_col=3, shuffle=False, cols=[0,1,3], keep_dtypes=True)
        0    1   3
    0   1   82  91
    1  72   56  76
    2  68   35  63
    3  64   65  22
    4   1   31  73
    5  69   79  71
    6  82   61  92
    7  -6   14  36
    8  30    6  40
    9   9  102  81

    References
    ----------
    [1] Chamikara, M. A. P., Bertók, P., Liu, D., Camtepe, S., & Khalil, I. (2020). Efficient privacy preservation of big data for accurate data mining. Information Sciences, 527, 420-443. https://arxiv.org/pdf/1906.08149.pdf
    [2] https://github.com/chamikara1986/PABIDOT/blob/master/PABIDOT.m
    """
    if sensitive_col is None:
        sensitive_col = []
    else:
        sensitive_col = [sensitive_col]

    if perturb_cols is None:
        perturb_cols = [col for col in data.columns if col not in sensitive_col]

    if not all([col in data.columns for col in [*perturb_cols, *sensitive_col]]):
        raise InputError(
            f"Values of `perturb_cols` and/or `sensitive_col` are not in `data` columns: {list(data.columns)}. (Received: {perturb_cols} and {sensitive_col}, respectively)"
        )

    if set(perturb_cols).intersection(sensitive_col):
        raise InputError(
            f"Values of `perturb_cols ` and `sensitive_col` must not overlap. (Received: {perturb_cols} and {sensitive_col}, respectively)"
        )

    if not is_numeric_dtype(data[perturb_cols].values):
        raise InputError(
            f"Columns to be transformed can only be numeric. (Received: {perturb_cols})"
        )

    n_pc = len(perturb_cols)

    perturb_cols = [c for c in perturb_cols if not _is_identical(data[c])]
    if len(perturb_cols) != n_pc:
        if len(perturb_cols) == 0:
            raise InputError(
                f"The values of `data[{perturb_cols}]` are all identical and therefore cannot be used for correlation."
            )
        else:
            print(
                "WARNING: ignoring columns that are composed entirely of identical values."
            )
    elif len(perturb_cols) == 1:
        raise InputError(
            f"The length of `perturb_cols` must be greater than 1. (Received: {len(perturb_cols)})"
        )

    z = stats.zscore(data[perturb_cols])
    cov = np.corrcoef(z.T)
    ncols = data[perturb_cols].shape[1]

    coords = _get_coordinates(ncols)

    rot_params = _gen_rotation_parameters(ncols, coords, cov)

    # Retrieving optimal theta and optimal axis of reflection
    i = np.argmin(rot_params.T, axis=0)  # idx
    m = np.take_along_axis(rot_params, np.expand_dims(i, axis=-1), axis=-1).squeeze()

    best_theta = np.argmax(m)  # optimal privacy guarantee
    refaxis = i[best_theta]

    # Generating the reflection matrix
    id_ref = np.eye(ncols)
    id_ref[:, refaxis] = -id_ref[:, refaxis]

    # Applying reflection transformation to data
    bo = (id_ref @ z.T).T

    # Translation Matrix Generation/Application
    idtrans = np.eye(ncols + 1)  # add a new row for the homogeneous coordinate
    idtrans[:ncols, ncols:] = np.random.uniform(size=(ncols, 1))

    # multidim translations; adding ones column for homogeneous coordinate
    multitrans = np.concatenate((bo, np.ones(shape=(bo.shape[0], 1))), axis=1)
    transresult = (idtrans @ multitrans.T).T
    # removing homogeneous coordinate
    multitrans = transresult[: bo.shape[0], : bo.shape[1]]

    bo = multitrans

    rad_value = best_theta * np.pi / 180.0

    # Generate the optimal rotational matrix and application of rotational transformation
    opt_rot_plane = _get_rotation_plane(ncols, coords, rad_value)
    bo = opt_rot_plane @ bo.T

    # Randomized expansion
    sign = np.sign(bo)
    bo = np.add(abs(bo), abs(np.random.uniform(size=bo.shape) * magnitude))
    bo = (bo * sign).T
    bo = bo * data[perturb_cols].std().values + data[perturb_cols].mean().values

    shuff_idx = data.index
    if shuffle:
        shuff_idx = np.random.choice(
            range(bo.shape[0]), size=(bo.shape[0]), replace=False
        )

    data.loc[:, perturb_cols] = bo[
        shuff_idx,
    ]
    if len(sensitive_col) != 0:
        data.loc[:, sensitive_col] = data.loc[shuff_idx, sensitive_col].reset_index(
            drop=True
        )
    else:
        data = data.iloc[shuff_idx].reset_index(drop=True)

    return data
