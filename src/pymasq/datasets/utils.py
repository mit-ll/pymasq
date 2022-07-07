from random import sample, choice
from numpy import ndarray

from pymasq import BEARTYPE


@BEARTYPE
def rand_cat_change(arr: ndarray, p: float) -> ndarray:
    """
    Randomly changes a portion of a categorical array to a different category
    in the array

    Parameters
    ----------
    arr : array-like
        An array of strings or categories

    p : float
        Proportion of the array to change categories, between 0.0 and 1.0

    Returns
    -------
    array-like
        An array with a user-defined portion of the elements randomly changed
    """
    if not 0 <= p <= 1:
        # TODO: raise optimizations.NotInRangeError
        raise ValueError("p must be between 0.0 and 1.0")
    a = arr.copy()
    cats = list(set(a))
    if len(cats) <= 1:
        raise ValueError("array must contain more than 1 category")
    class_dict = {c: [i for i in cats if i != c] for c in cats}
    n = len(a)
    elements = sample(range(0, n), int(n * p))
    for e in elements:
        a[e] = choice(class_dict[a[e]])
    return a
