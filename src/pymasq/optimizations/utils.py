from typing import Tuple
import pandas as pd
from typing import Callable, Dict, List, Optional, Union

from pymasq import BEARTYPE
from pymasq.optimizations import IterativeSearch


@BEARTYPE
def apply_and_evaluate(
    target: pd.DataFrame,
    mutations: List[Dict],
    metrics: Dict[Union[str, Callable], Dict],
    verbose: int = 0,
) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
    """This helper method calls IterativeSearch in a specific fashion:
    1) The given mitigation list is applied in order, without reuse or randomization
    2) The given metrics are applied to the modified dataframe returned by step one.
    In step 1, no metrics are applied. In step 2 no mitigations are applied. In other words,
    the metrics are calculated once only: after all mitigations have been applied.

    Parameters
    ----------

    target : pandas.DataFrame
        The dataframe that will be manipulated and evaluated.

    mutations : List[Dict]
        List of mutations (e.g., mitigations) to use when manipulating
        the `target` dataframe. Each mutation should be defined as a
        separate `dict` in the `list`, where the two top-level keys are the
        names of the `pymasq` mitigation to use or the function itself
        (e.g., `"add_noise"` or `pymasq.mitigations.add_noise`), and
        `p`, which defines the probability of choosing that specific mitigation.

        The values of the function keys must also be a `dict` with the key-value pairs
        that will parameterize each respective mitigation function.

        The values of each probability key must be an `int` or `float`.
        For `float` values, each probability should be rounded to 2 decimal places
        and the sum of each probability must sum to 1. Note that an `int`
        value will imply that there is only 1 mitigation to be used.

        Note that mitigations are applied even if they reduce the fitness score calculated
        by the given metric.

    metrics : Dict[Union[str, Callable], Dict]
        Functions used to evaluate `target` dataframe. The top-level keys should
        be the names of the `pymasq` metric to use or the function itself (e.g.,
        `"auc_score"` or `pymasq.metrics.auc_score`).

        The values of the function keys must also be a `dict` with `weight` and
        `params` as keys. The weight key's value must be a float, and the `params`
        key's value must be a `dict` with the key-value pairs that will parameterize
        each respective metrics function.

    verbose : int
        Set the verbosity level throughout the optimization procedure. (Default: 0)

    Returns
    -------
    pd.DataFrame:
        The modified target
    float:
        The fitness value
    list:
        The log of mutations applied

    Example
    -------
    >>> from pymasq.datasets import load_census
    >>> from pymasq.utils import apply_and_evaluate
    >>> from pymasq.metrics import k_anon, propensity_score

    Load the census dataset

    >>> df = load_census()

    Set the evaluation function to be the k-anonymity risk metric and set
    its respective parameters.

    >>> eval_fxns = {
        k_anon: {"params": {"key_vars": ["sex"]}},
        propensity_score: {"params": {"sensitive_col": "age","method":"larscv","preprocessor":"label_encode"}}
        }

    Set the mitigation functions to be applied.

    >>> mit_fxns = [
        {"hashing": {"cols": ["education"]}},
        {"add_noise": {"cols": ["age"],"method":"additive", "magnitude": 100}}
       ]

    Execute.

    >>> mod_df, fitness, log = apply_and_evaluate(df, mit_fxns, eval_fxns)
    """

    # lambda metric that returns 1.0 always
    ones = {lambda: 1.0: {}}

    # setup and call Iterative with the trivial metric
    no_metrics = IterativeSearch(
        iters=len(mutations),
        target=target,
        mutations=mutations,
        reuse_mutations=False,
        randomize_mutations=False,
        metrics=ones,
        verbose=verbose,
    )
    data_mod, _, _ = no_metrics.optimize()

    # take the returned data_mod and apply metric, without iterations
    no_mitigations = IterativeSearch(
        iters=0,
        target=data_mod,
        mutations=[{}],
        reuse_mutations=False,
        randomize_mutations=False,
        metrics=metrics,
        verbose=verbose,
    )
    _, fitness, log = no_mitigations.optimize()

    return (data_mod, fitness, log)
