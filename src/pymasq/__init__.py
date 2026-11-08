from os import path

__version__ = "1.0.1"


try:
    # import the runtime type checker, if available
    from warnings import filterwarnings

    from beartype import beartype
    from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

    filterwarnings(action="ignore", category=BeartypeDecorHintPep585DeprecationWarning)
    BEARTYPE = beartype
except ImportError:
    BEARTYPE = lambda func: func


def set_seed(r=None):
    """Set a module-wide random seed.

    Typically this would be used for testing purposes
    and replicating experiments.
    """
    import random
    import struct
    from os import urandom

    import numpy as np

    if r is None:
        r = struct.unpack("i", urandom(4))[0]
    elif isinstance(r, (tuple, list)):
        r = r[0]

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
    "models",
    "optimizations",
    "preprocessing",
    "plotting",
    "sa",
    "utils",
    "config",
]
