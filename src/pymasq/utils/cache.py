import glob
import hashlib
import hmac
import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

import pymasq.config as cfg
from pymasq import BEARTYPE
from pymasq.errors import InputError

logger = logging.getLogger(__name__)

def _hmac(data: Union[bytes, str]) -> str:
    """
    Returns the SHA-256 HMAC of given data as a string
    """
    if isinstance(data, str):
        data = data.encode()
    return hmac.new(cfg.CACHE_HMAC_KEY.encode(), data, hashlib.sha256).hexdigest()


def _verify_hmac(data: bytes, expected_digest: str) -> bool:
    """
    Verifies data against an expected HMAC digest using timing-safe comparison.
    Supports SHA-256 and MD5 (for legacy caches).
    """
    digest_sha = hmac.new(cfg.CACHE_HMAC_KEY.encode(), data, hashlib.sha256).hexdigest()
    if hmac.compare_digest(digest_sha, expected_digest):
        return True
    digest_md5 = hmac.new(cfg.CACHE_HMAC_KEY.encode(), data, hashlib.md5).hexdigest()
    return hmac.compare_digest(digest_md5, expected_digest)


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
    cache_location.mkdir(parents=True, exist_ok=True)
    try:
        total, _, free = shutil.disk_usage(cache_location)
        percent_free = free / total
        if percent_free < cfg.FREE_SPACE_THRESHOLD:
            for cached_file in cache_location.glob("*.pkl"):
                try:
                    cached_file.unlink()
                except OSError:
                    pass
    except OSError:
        pass

    pickled_data = pickle.dumps([description, data])

    filename = f"{fn_prefix}.{_hmac(pickled_data)}.pkl"
    if verbose > 0:
        logger.info(f"Saving. hmac key is: {cfg.CACHE_HMAC_KEY}")
    with open(filename, "wb") as fd:
        fd.write(pickled_data)

    with open(filename, "rb") as fd:
        assert _verify_hmac(fd.read(), _hmac(pickled_data))
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
        try:
            with open(file, "rb") as fd:
                file_bytes = fd.read()
        except OSError:
            continue

        file_digest = file.split(".")[-2]
        # check the hmac of the file (unless ignore)
        if not ignore_hmac and not _verify_hmac(file_bytes, file_digest):
            if verbose > 0:
                logger.info(
                    f"Error: HMAC mismatch for file {file} with hmac key of '{cfg.CACHE_HMAC_KEY}'"
                )
            continue
        if verbose > 0:
            logger.info(f"Filename hmac verified: {file_digest}")

        # read in the data
        try:
            description, data = pickle.loads(file_bytes)
            if verbose > 0:
                logger.info(f"{description}")
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
    logger.info(f"Checking all files in {file_or_path}")
    result = {}
    for file in glob.glob(file_or_path + "/*.pkl"):
        logger.info(f"\n----{file}----")
        try:
            _, description = load_cache(prefix_path=file)
        except Exception as e:
            logger.info(e)
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

    row_hashes = np.sort(hash_pandas_object(df).to_numpy())
    file_name = hashlib.sha256(row_hashes.tobytes()).hexdigest()
    filename_with_path = os.path.join(path, f"{file_name}.{suffix}")
    return filename_with_path
