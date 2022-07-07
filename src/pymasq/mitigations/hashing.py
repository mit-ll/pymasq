from typing import Any, Callable, Dict, List, Optional, Union

import hashlib
import numpy as np
import os
import pandas as pd

from pymasq import BEARTYPE
from pymasq.config import (
    FORMATTING_ON_OUTPUT,
    FORMATTING_IGNORE_DTYPES,
)
from pymasq.errors import InputError
from pymasq.utils import formatting

__all__ = ["hashing"]


@formatting(on_output=FORMATTING_ON_OUTPUT, ignore_dtypes=True)
@BEARTYPE
def hashing(
    data: Union[pd.Series, pd.DataFrame],
    hash_func: Union[Callable, str] = "sha256",
    salt: Optional[Union[List, np.ndarray, str, int]] = None,
    append_salt: bool = True,
    store_salt: Optional[str] = None,
    cols: Optional[Union[List, str, int]] = None,
) -> pd.DataFrame:
    """Map all input values onto a set of output values using a hashing function.

    Parameters
    ----------
    data : DataFrame, Series, or array_like
        The data to be modified.
    hash_func : function or str (Default: "sha256")
        The hashing function to use. If `hash_func` is a string, then it should correspond to
        a function name in the `hashlib` Python library [1]_. Else, it will apply the user-defined function.
        Algorithms listed in `hashlib.algorithms_guaranteed` are prefererd.
    salt : list, str, or int, Optional
        The salt, or random data, to add to `data` to perturb it before hashing occurs. 
        If left as `None`, then no salt will be added to `data`. If `salt` is a list, 
        then it must be of the same length as `data`. If `salt` is a string, then the same salt value 
        will be added to each value in `data`. If `salt` is an integer, then a random salt of that bit size 
        will automatically be generated (note that 16 and 32 are typical salt bit sizes).
        Generated salts can be stored by specifying the `store_salts` parameter.
        Please refer to [2]_ for additional information on the importance of salts.
    append_salt : bool (Default: True)
        If True, the salt will be appended at the end of `data`. Otherwise, the salt will be prepended to the front of `data`.
    store_salt : str, Optional
        The filename or complete file path where salts will be stored to. The `salt` will automatically
        be saved as a CSV file. Note that if `salt` is an integer, salt values will be generated dynamically, therefore,
        storing the salt values will become the only method of recovering the original value of `data`.
        Any file of the same name as `store_salt` will be overwritten.
    cols : str or list, Optional
        The name of the column or columns to subset from `data` if `data` is a dataframe.

    Returns
    -------
    DataFrame
        A DataFrame with hashed values.

    Notes
    -----
    Use `hashlib.algorithms_available` to see available hashing functions.

    References
    ----------
    https://docs.python.org/3/library/hashlib.html

    https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-132.pdf

    Examples
    --------
    >>> from pymasq.datasets import load_census
    >>> df = load_census()
    >>> df = df.loc[:10, ["marital_status", "occupation"]]
    >>> df
        marital_status         occupation
    0   Never-married          Adm-clerical
    1   Married-civ-spouse     Exec-managerial
    2   Divorced               Handlers-cleaners
    3   Married-civ-spouse     Handlers-cleaners
    4   Married-civ-spouse     Prof-specialty
    5   Married-civ-spouse     Exec-managerial
    6   Married-spouse-absent  Other-service
    7   Married-civ-spouse     Exec-managerial
    8   Never-married          Prof-specialty
    9   Married-civ-spouse     Exec-managerial
    10  Married-civ-spouse     Exec-managerial

    >>> hashing(df, hash_func="md5")
        marital_status                    occupation
    0   73cf421158c109aa4db6861489471f81  4c4e21b293a987859a0d20bcba98157f
    1   cbd21dd3a341657f6d347dfed4717deb  8f2cf8dfc926906b968e7f5e0588f55a
    2   f2e4016d0c9314f9cf4b36489da5b0dd  379f2522eec3badf2b0c23ee58a60a72
    3   cbd21dd3a341657f6d347dfed4717deb  379f2522eec3badf2b0c23ee58a60a72
    4   cbd21dd3a341657f6d347dfed4717deb  e382d0ae524f4f8dbb7f3e320e05b600
    5   cbd21dd3a341657f6d347dfed4717deb  8f2cf8dfc926906b968e7f5e0588f55a
    6   538a9356eea33950d99104606fbe09ad  c730c4fd37611c3afc670d4c524ec180
    7   cbd21dd3a341657f6d347dfed4717deb  8f2cf8dfc926906b968e7f5e0588f55a
    8   73cf421158c109aa4db6861489471f81  e382d0ae524f4f8dbb7f3e320e05b600
    9   cbd21dd3a341657f6d347dfed4717deb  8f2cf8dfc926906b968e7f5e0588f55a
    10  cbd21dd3a341657f6d347dfed4717deb  8f2cf8dfc926906b968e7f5e0588f55a

    >>> hashing(df, hash_func="md5", salt="random-string")
        marital_status	                    occupation
    0	7a14ab6033c824ac361a1794febd2e19	21c32d969ed3c02e4608f90262f3884c
    1	386dda8775ed53032c7e5a84d2f18399	91a3639f58d9711bd15436a09af4fa73
    2	38aec33b4b1518f1949f4e8e98b8cf89	adb1a3c463d9a7a1525efe6aee49413f
    3	386dda8775ed53032c7e5a84d2f18399	adb1a3c463d9a7a1525efe6aee49413f
    4	386dda8775ed53032c7e5a84d2f18399	4538d56cfcbd690a2d2eea3e557bb48f
    5	386dda8775ed53032c7e5a84d2f18399	91a3639f58d9711bd15436a09af4fa73
    6	8e378f2aaa2a37999eb51aded134048f	87653015c652de3cefec557b5497b00f
    7	386dda8775ed53032c7e5a84d2f18399	91a3639f58d9711bd15436a09af4fa73
    8	7a14ab6033c824ac361a1794febd2e19	4538d56cfcbd690a2d2eea3e557bb48f
    9	386dda8775ed53032c7e5a84d2f18399	91a3639f58d9711bd15436a09af4fa73
    10	386dda8775ed53032c7e5a84d2f18399	91a3639f58d9711bd15436a09af4fa73

    >>> hashing(df, hash_func="md5", salt=16)
        marital_status                      occupation
    0	c9d3a62a96acd378954cc8928f716259	b989a09703027029b83c8b0f5ccb73dd
    1	625903e06f5cce02129e7e2b6cbb6b27	9fb4f8f0f4c98da85e328b0d52fc5bdf
    2	e0b2e2202d92e0b409be4a3fea83ef95	9b2b47e32d239641442089ce1af41342
    3	3e13c0bce44e7615a964ff531e207ccd	b3df77e0c7cade94afdd6d67fee5dddf
    4	8820d08088724128c99a2b3aaee308a1	4208add69e4bfdb506912a5ad2c7be48
    5	588cdaef14ecbf2e4423f2c61f0ebc08	49af370931c3ee27a38a74b6b6587fa8
    6	60a94bd6425d2746a1fee6a4bc79314b	9ef0a203e97dc1ae6c19497ccabfcbc2
    7	9c34c01b6f20e319340be9a04024a00b	01635612db0afbcf86b97b4744c16610
    8	157acb7a5bf8cb2a5f8acb49f3f726c4	41c50586e58d07257c23ab605842fe14
    9	a8a4bcf40ed192185455e17473105f8d	70ddd4700872cc81f0d58f0039762794
    10	0806d9fbcd3a12c2503a8dda7622a765	e3fc14fc7f8e5ecd6c1e220b0fd8b686

    >>> hashing(df, hash_func="md5", salt=np.random.choice(["a","b","c"], size=df.shape).tolist())
        marital_status	                    occupation
    0	d3c2c07328c004257d78078efc18a2e7	dfd8c82cf825bb141364719be36577e1
    1	34f4e90dd6cd3c8e0c9be987bcb890a2	b63bd8fdcbdff01408a6f037e1816e69
    2	bb96b852df8daef0c978f8191b7ab353	50f7a90fc49880975fdc3f598deeb064
    3	34f4e90dd6cd3c8e0c9be987bcb890a2	3c3b7950a48c64e0123042f791e5abc6
    4	b3c70e439b5c9ec15c91e2624dbe5c71	a3157d1201d17d2da30f224f99e56730
    5	b3c70e439b5c9ec15c91e2624dbe5c71	94798b137db818bf2213fcabb99add4d
    6	5c9747c0f0d0b7029a572acf617ba46a	0bd722b7a84b3426ff022939035be2c0
    7	4222ef367e33c9d84a585024c56e5622	49ec30c27fbea6bb4ab691e51e598c76
    8	d3c2c07328c004257d78078efc18a2e7	0c95c9bcba4f3483df0fe58da41c1624
    9	34f4e90dd6cd3c8e0c9be987bcb890a2	94798b137db818bf2213fcabb99add4d
    10	34f4e90dd6cd3c8e0c9be987bcb890a2	49ec30c27fbea6bb4ab691e51e598c76

    >>> hashing(df, salt=16, store_salt="my_salt.csv")
        marital_status	                                    occupation
    0	f99438d80007788849824e3cff625cd43a653028ee029e...	327f8dc400dba2d4c0aa3b78d2c274efd7060e8d1e084e...
    1	79c8e76b8c0240a4dba439e436f4325a9d56bdc4b8ccfa...	7a3501b7a9d7833c05efd1b691d08361e6a4225e9ec87e...
    2	39b069e5943d6f5fdd9775b1fb7ab52173a83e9ad58bcc...	89f5df3961bb6650fef80549933eb0d43dca5218d35d3a...
    3	9d74acf0c0d4627b07643156f36d217e6e65cf57ed650c...	4eb952b7fc89afb7c0e4771986a2a6ec437675f2356077...
    4	c2d922e99744ab86e22babd8ec9275c3222d6aee699179...	2ffe7708a15b0ade8788172d52e793b570b6ca4a420393...
    5	b9c5f6a5300e2985f231e04136b30d91d3d02a89e7b580...	b7d5b50b8ade16b62515cbbe65e8e1a28ced946238fc74...
    6	4ac1a2f5bb91cafee4f078ac45e52d811a3ab26b248eb0...	7b6ec0e5b1f1ea59abc9493b15cb585b1e5037d12a5c13...
    7	24d88b2478f1a78df470304a1433f9ee5052c0085aeed8...	40f638562b4b37b250973cd09759d95473d63aeadf75f9...
    8	9856ef4eaefd2f6316c7e53df803f84373adaba477dc88...	f04851e03e9ab298d5103a03bbe959de2501538df585e0...
    9	4a84196a0a0ae0aca9611d612ce1e2153c03644449c44e...	8a394ba22d4eaaf9ff6a2ca8260b024c7e06ebdc1e839c...
    10	89f3b96a56971718c5db877bdafcb83c0747bab4e7a682...	9ba3f0fb9d6d363f98b47875451954f4d41853281327dd...
    """
    data = data.astype(bytes)

    if salt is not None:
        salt_df = pd.DataFrame(index=data.index, columns=data.columns, dtype=bytes)
        if isinstance(salt, (np.ndarray, list)):
            salt_df = pd.DataFrame(
                salt, index=data.index, columns=data.columns, dtype=bytes
            )
            if salt_df.shape != data.shape:
                raise InputError(f"Incorrect `salt` dimensions; expected {data.shape}. (Received: {salt_df.shape})")
        elif isinstance(salt, str):
            salt_df[:] = salt.encode()
        elif isinstance(salt, int):
            salt_df = salt_df.applymap(lambda v: os.urandom(salt))
        else:
            raise InputError(f"Invalid `salt` type; only types allowed are `list`, `str`, and `int`. (Received: {type(salt)})")

        data = (data + salt_df) if append_salt else (salt_df + data)

        if store_salt is not None:
            with open(store_salt, "w") as f:
                salt_df.to_csv(f, index=False)

    if isinstance(hash_func, Callable):
        return data.applymap(lambda v: hash_func(v))

    hash_func = getattr(hashlib, hash_func)

    if "shake" in str(hash_func):
        # TODO: change to logging
        print(f"Warning: the default length of the hexdigest is set to 16; to alter the length, pass in `{hash_func}` as a callable defined with your prefered length.")
        return data.applymap(lambda v: hash_func(v).hexdigest(16))
    
    return data.applymap(lambda v: hash_func(v).hexdigest())
