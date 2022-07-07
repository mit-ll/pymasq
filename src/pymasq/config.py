from pathlib import Path
from typing import Tuple
from pymasq import ROOT_DIR

# Directory where all embeddings and models will be cached
CACHE_LOCATION: Path = Path("~/.cache/pymasq").expanduser()

CACHE_HMAC_KEY: str = "To ensure integrity of files cached by pymasq, change this string to something unique and unguessable."

# If proportion of free space goes below this threshold, then the cache will be deleted
FREE_SPACE_THRESHOLD: float = 0.05

# cast the output of mitigations back to input type (e.g., input series, return series)
FORMATTING_ON_OUTPUT: bool = False

# cast the values of a column back to their original dtype (e.g., input integer-column, return integer-column)
FORMATTING_IGNORE_DTYPES: bool = False

# validation of numeric values
VALIDATE_NUMERIC_ON_INPUT: bool = False
VALIDATE_NUMERIC_ON_OUTPUT: bool = False

# Types of Regressor models
REGRESSOR_MODELS: Tuple[str] = ("encv", "rfreg", "tpotreg", "larscv")

# Types of Classifier models
CLASSIFIER_MODELS: Tuple[str] = ("logreg", "rfclass", "tpotclass")

DEFAULT_LOGISITIC_REGRESSION_SOLVER: str = "saga"

# Byte Pair Encoding default language and dimensionality for vectors
BPE_LANG: str = "en"
BPE_DIM: int = 50

# Maxnimum number of unique categories that defines a categorical vs. a string column
CATEGORY_THRESHOLD: int = 30

# Default random seed for reproducibility
DEFAULT_SEED: int = 1234

# Default number of parallel processors, set to -1 for all processors
DEFAULT_N_JOBS: int = -1
