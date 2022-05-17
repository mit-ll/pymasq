from os import path

__version__ = "0.6.5"


try:
    # import the runtime type checker, if available
    from beartype import beartype

    BEARTYPE = beartype
except ImportError:
    BEARTYPE = lambda func: func


def set_seed(r=None):
    """Set a module-wide random seed

    Typically this would be used for testing purposes
    and replicating experiments.
    """
    import numpy as np
    import struct
    import random
    from os import urandom

    if r is None:
        r = struct.unpack("i", urandom(4))

    _random_seed = int(r)
    np.random.seed(_random_seed)
    random.seed(_random_seed)


def new_token():
    """Returns a unique identifier string.

    Returns
    -------
    token : string
        Unique identifier combined from datetime metadata.
    """
    from datetime import datetime

    time_stamp = datetime.now()
    token = "%s%s%s_%s%s%s" % (
        time_stamp.year,
        time_stamp.month,
        time_stamp.day,
        time_stamp.hour,
        time_stamp.minute,
        time_stamp.second,
    )

    return token


ROOT_DIR = path.dirname(path.abspath(__file__))

TOKEN = new_token()


__all__ = [
    # package var
    "ROOT_DIR",
    "TOKEN",
    # subpackages
    "datasets",
    "kve",
    "metrics",
    "mitigations",
    "optimizations",
    "preprocessing",
    "plotting",
    "config",
]
