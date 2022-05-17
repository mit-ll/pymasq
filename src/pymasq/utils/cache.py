import pickle
import hashlib
from pathlib import Path
import hmac
import glob
import shutil
from typing import Optional, Tuple, Dict, Union
from pandas.util import hash_pandas_object
import pandas as pd
from pymasq import BEARTYPE
import pymasq.config as cfg
from pymasq.errors import InputError
import os


def _hmac(data: object) -> str:
    """
    Returns the md5 hash of given data as a string
    """
    return hmac.new(str.encode(cfg.CACHE_HMAC_KEY), data, hashlib.md5).hexdigest()


@BEARTYPE
def save(
    data: object,
    cache_location: Union[str, Path],
    fn_prefix: Union[str, Path],
    description: Optional[str] = None,
    verbose: int = 0,
):
    """
    This method saves the already trained model of the given dataframe to a cached pickle file.
    The hmac of the pickled data (including the description) becomes the suffix of cached filename. The key for the hmac is
    stored in pymasq.config.CACHE_HMAC_KEY.

    Parameters
    ----------
    data: object
        A data accepted by pickle
    cache_location: str
        The cache_location of the file
    fn_prefix: str
        This is the prefix of the cached filename. The suffix will be based on an hmac
    verbose: int, optional (Default:0)
        0 disables verbose logging
        1 (or higher) enables verbose logging
    descriptions: str, optional (Default: None)
        A string saved with the pickle file.

    Returns
    -------
    str
        Name of the cached file
    """
    cache_location = Path(cache_location)
    total, _, free = shutil.disk_usage(cache_location)
    percent_free = free / total
    if percent_free < cfg.FREE_SPACE_THRESHOLD:
        shutil.rmtree(cache_location)
    cache_location.mkdir(parents=True, exist_ok=True)

    pickled_data = pickle.dumps([description, data])

    filename = f"{fn_prefix}.{_hmac(pickled_data)}.pkl"
    if verbose > 0:
        print(f"Saving. hmac key is: {cfg.CACHE_HMAC_KEY}")
    with open(filename, "wb") as fd:
        fd.write(pickled_data)

    assert _hmac(open(filename, "rb").read()) == _hmac(pickled_data)
    return filename


@BEARTYPE
def load_cache(
    prefix_path: Union[str, Path],
    verbose: int = 0,
    ignore_hmac: bool = False,
) -> Tuple[Optional[object], Optional[str]]:
    """
    This method loads the cached pickle file of trained classifier from df to self.trained.
    It checks the contents of the file against the hmac stored in the filename, and it will
    not load unless the hmac is valid (Unless ignore_hmac is True).  The key for the hmac is
    stored in pymasq.config.CACHE_HMAC_KEY.

    In pymasq, often the cache file represents a model trained from a dataframe. The hash of the
    dataframe is the prefix of the cache file. The suffix is the hmac of the trained model. Therefore,
    when a cache is loaded, the suffix of the file is unknown to the method calling load_cache(). Therefore
    this method attempts to load all cache files that have the prefix. Mismatching hmacs do not raise an
    exception and instead simply print an error. Cache files that have a valid hmac but cannot be loaded
    do raise an exception.

    Parameters
    ----------
    prefix_path: str
        The path to a cached file to be loaded.
    verbose: int, optional (Default: 0)
        0 disables verbose logging
        1 (or higher) enables verbose logging
    ignore_hmac: book (Defaut: False)
        If true, a failed hmac is ignored and the data is loaded (which is not a secure option).
    Returns
    -------
    object:
        The stored object
    str:
        A description that was stored with the object when it was saved.
    """

    for file in glob.glob(str(prefix_path) + "*"):
        digest = _hmac(open(file, "rb").read())
        # check the hmac of the file (unless ignore)
        if str(digest) != file.split(".")[-2] and not ignore_hmac:
            if verbose > 0:
                print(
                    f""""
                    Error: hmac of file ({digest}) does not match the hmac stored in the filename 
                    ({file.split('.')[-2]}) for hmac key of '{cfg.CACHE_HMAC_KEY}' for file: {file}
                    """
                )
            continue
        if verbose > 0:
            print(f"Expected hmac: {str(digest)}")
            print(f"Filename hmac: {file.split('.')[-2]}")

        # read in the data
        try:
            fd = open(file, "rb")
            description, data = pickle.load(fd)
            fd.close()
            if verbose > 0:
                print(f"{description}")
            return data, description
        except Exception as e:
            raise InputError(f"Error loading cache file from {prefix_path}: {e}")
    # no file found or matched hmac
    return None, None


@BEARTYPE
def cache_info(file_or_path: str) -> Dict[str, str]:
    """
    Checks the hmac of a stored file (or directory of files) and then prints the
    descriptive header stored in it (if any). The descriptions are printed only
    if the hmac of the file matches the one generated using pymasq.confif.hm

    Parameters
    ----------
    file_or_path: str
        The full path to a specific file, or if a directory is given.
        All files ending in ".pkl" are examined, but only those with a valid
        hmac are loaded.

    Returns
    -------
    Dict[str,str]
        str:
            the name of the file in storage
        str:
            The description stored with the file
    Files without valid hmacs are not listed.

    """
    print("Checking all files in ", file_or_path)
    result = {}
    for file in glob.glob(file_or_path + "/*.pkl"):
        print(f"\n----{file}----")
        try:
            _, description = load_cache(prefix_path=file)
        except Exception as e:
            print(e)
            continue
        if description is not None:
            result[file] = description
    return result


@BEARTYPE
def df_to_filename_prefix(
    df: pd.DataFrame,
    suffix: str,
    path: Union[str, Path] = ".",
) -> str:
    """
    This method produces a consistent file name (including path) for a given DF. Note that two data frames that differ
    only in the order of their rows, rather than the content of the rows, will have the same result.

    Parameters
    ----------
    df: pd.Dataframe
        A dataframe.
    path: cache location
    suffix: name

    Returns
    --------
    str:
        The name of the file, including the path.
        The dataframe, `df`, is not modified.
    """

    df_as_list = [str(x) for x in sorted(hash_pandas_object(df))]
    df_as_str = "".join(df_as_list)
    df_as_bytes = bytes(df_as_str.encode())
    file_name = hashlib.md5(df_as_bytes).hexdigest()
    filename_with_path = os.path.join(path, f"{file_name}.{suffix}")
    return filename_with_path
