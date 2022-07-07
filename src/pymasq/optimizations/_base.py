import copy
import inspect
import numpy as np
import pandas as pd
from abc import abstractmethod

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# import pymasq
from pymasq import BEARTYPE
import pymasq.mitigations as mits
import pymasq.metrics as mets

from pymasq.errors import (
    SumNotEqualToOneError,
    NotInRangeError,
    LessThanZeroError,
    NoMutationAvailableError,
)
import sys


class OptimizationBase:
    """Base class for the optimization algorithms.

    The `pymasq.optimization` search procedures enable automated exploration and evaluation
    of different mitigation strategies. At each iteration, the optimization procedures will
    perform two functions: a mutation and an evaluation. The former will alter current
    target dataframe using one of the mitigations specified. The latter will evaluate the
    fitness of the current target dataframe with respect to the risk and/or utility metrics
    selected. The objective of the optimization procedures is to minimize the weighted
    average fitness of the evaluation metrics over each iteration.

    The optimization procedures will terminate once they reach a set number of iterations,
    or the fitness value of the target datfarme falls below a specified threshold, `theta`.
    Additional stopping criteria are available for select optimization procedures.

    Note that the result of each search procedure can vary, since it is optimizing only with respect
    to the parameters specified (e.g., mitigations and metrics, along with their associated parameters).

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

    metrics : Dict[Union[str, Callable], Dict]
        Functions used to evaluate `target` dataframe. The top-level keys should
        be the names of the `pymasq` metric to use or the function itself (e.g.,
        `"auc_score"` or `pymasq.metrics.auc_score`).

        The values of the function keys must also be a `dict` with `weight` and
        `params` as keys. The weight key's value must be a float, and the `params`
        key's value must be a `dict` with the key-value pairs that will parameterize
        each respective metrics function.

    iters : int
        Number of iterations to run optimization procedure for. (Default: 10)

    reuse_mutations : bool
        Do not discard mutations after being used. (default: True)

    randomize_mutations : bool
        Select mutations at random. If False, they will be applied
        in the order they were specified and `reuse_mutations` will be
        set to `False`. (Default: True)

    theta : float
        Threshold value to reach. If reached, the optimization will stop. (Default: 0.0)

    verbose : int
        Set the verbosity level throughout the optimization procedure. (Default: 0)

    headers : List, Optional
        Additional column names to include in each logbook. These
        will vary based on the algorithm being used.
    """

    @BEARTYPE
    def __init__(
        self,
        target: pd.DataFrame,
        mutations: List[Dict],
        metrics: Dict[Union[str, Callable], Dict],
        headers: Optional[List[str]] = None,
        iters: int = 10,
        reuse_mutations: bool = True,
        randomize_mutations: bool = True,
        theta: Union[int, float] = 0.0,
        verbose: int = 0,
        exit_on_error: bool = True,  # Don't change to False without considering impact on pytests.
        **kwargs,
    ):

        self.target = target
        self.mutations = mutations
        self.metrics = metrics
        self.iters = iters
        self.theta = theta
        self.randomize_mutations = randomize_mutations
        self.reuse_mutations = reuse_mutations if randomize_mutations else False
        self.verbose = verbose
        self._headers = list(metrics.keys()) + (headers or [])
        self._logbook = None
        self.exit_on_error = exit_on_error

        self.progress_reporter = kwargs.get("callback", None)
        if self.progress_reporter:
            self.progress_reporter(0.0)

        if iters < 0:
            raise LessThanZeroError(f"Iterations must be >= 0. (Received: {iters})")
        if not 0.0 <= theta <= 1.0:
            raise NotInRangeError(
                f"Threshold (`theta`) must be between the interval [0,1], inclusive. (Received: {theta})"
            )

        probs = self._validate_input_sums(mutations, "p")
        if probs is None:
            raise ValueError(
                f"A probability `p` must be defined for each mutation in `mutations`. (Received: {mutations})."
            )
        prob_sum = sum(probs)
        if prob_sum == 0.0:
            probs = self._distribute(len(mutations))
            self.mutations = [
                dict(m, **{"p": probs[i]}) for i, m in enumerate(mutations)
            ]
        elif round(prob_sum, 5) != 1.0:
            raise SumNotEqualToOneError(
                f"Mitigation probabilities must sum to 1. (Received: {prob_sum})"
            )

        weights = self._validate_input_sums(list(metrics.values()), "weight")
        if weights is None:
            raise ValueError(
                f"An importance weighting `weight` must be defined for each metric in `metrics`. (Received: {metrics})"
            )
        weight_sum = sum(weights)
        if weight_sum == 0.0:
            weights = self._distribute(len(metrics))
            [
                v.update({"weight": weights[i]})
                for i, v in enumerate(self.metrics.values())
            ]
        elif weight_sum != 1.0:
            raise SumNotEqualToOneError(
                f"Metric importance weightings must sum to 1. (Received: {weight_sum})"
            )

        n_mutations = len(self.mutations)
        if not self.reuse_mutations and self.iters > n_mutations:
            self.iters = n_mutations
            if self.verbose:
                print(
                    ">>> [Info]: The number of iterations (%i)" % (iters),
                    "cannot exceed the number of mutations specified (%i)"
                    % (n_mutations),
                    "when `reuse_mutations` is False;",
                    "`iters` will be set to %i." % (n_mutations),
                )

        self._max_iters = self.iters

    @BEARTYPE
    def _validate_input_sums(
        self, values: List[Dict], key: str
    ) -> Optional[List[Union[int, float]]]:
        """Verify that each element in the input list dict has `key` defined or that None do.

        Parameters
        ----------
        values: List of dicts
            Input list of dicts to check for `key`.
        key : str
            The key to check in each element of `values`.

        Returns
        -------
        sums : List of int or floats
            . If no element has `key`, then `sums` will default a list of 0s of length `values`.
        """
        sums = []
        n_defined = 0.0
        for v in values:
            try:
                sums.append(v[key])
                n_defined += 1
            except KeyError:
                sums.append(0.0)
        # if n_defined == 0, then none were defined
        if n_defined != 0.0 and n_defined != len(values):
            # TODO: future iterations should distribute missing values and/or normalize
            return None
        return sums

    @BEARTYPE
    def _distribute(self, N: Union[int, float]) -> np.ndarray:
        magnitude = 100.0
        base, extra = divmod(magnitude, N)
        return np.array([base + (i < extra) for i in range(N)]) / magnitude

    # @BEARTYPE cannot handle `None` return without additional beartype imports; skipping type checking so beartype can remain optional
    def _record_stats(self, **kwargs: Dict[Any, Any]) -> None:
        """Record the statistics of a run in the Logbook.

        Parameters
        ----------
        kwargs: Dict[Any, Any]
            Values for algorithm-specific headers to be added to the `Logbook`.
        """
        self._logbook.update(kwargs)

    @BEARTYPE
    @abstractmethod
    def _optimize(self) -> Union[pd.DataFrame, float, pd.DataFrame]:
        pass

    @BEARTYPE
    def optimize(self) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
        """Run the optimization procedure.

        This procedure will vary based on which algorithm was chosen.

        Returns
        -------
        target : pd.DataFrame
            An altered dataframe with the best fitness value.

        fit : float
            The fitness value of the `target` dataframe.

        logbook : pd.DataFrame
            A dataframe with the records of each dataframe, mutation, and fitness value accross the optimization
        """
        if self.verbose:
            print("[Starting ...]")

        self._target = self.target.copy()
        self._iters = self.iters
        self._mutations = copy.deepcopy(self.mutations)
        self._logbook = Logbook(self._headers)

        target, fit, logbook = self._optimize()  # algo-specific

        if self.verbose:
            print("[... Search Complete]")

        if self.progress_reporter:
            self.progress_reporter(1.0)

        return target, fit, logbook.log

    def _safe_evaluate(
        self, target: pd.DataFrame
    ) -> Union[float, List[Tuple], List[Exception]]:
        """Adds a try/except wrapper to _evaluate"""
        try:
            fit, log = self._evaluate(target)
        except Exception as e:
            if self.exit_on_error:
                raise
            else:
                log = {}
                fit = np.inf
                error_log = [e]
        else:
            error_log = []
        return fit, log, error_log

    @BEARTYPE
    def _evaluate(self, target) -> Tuple[float, List[Tuple]]:
        """Evaluate the fitness of the `target` dataframe.

        The fitness value of a dataframe will be measured with the user-specified `metrics`
        and weighted accordingly.

        Parameters
        ----------
        target : pd.DataFrame
            The dataframe to be evaluated.

        Returns
        -------
        wfit : float
            Weighted average of the fitness values.

        fitnesses : List[Tuple[str, float, float, Dict]]
            List of the evaluation metrics used, including the function name, its returned value,
            its importance weighting, and its parameters.
        """
        fitnesses = []
        for func, args in self.metrics.items():
            if isinstance(func, str):
                func = getattr(mets, func)

            if self.verbose >= 2:
                print("\t[Evaluation]: %s" % (func))

            params = copy.deepcopy(args.get("params", {}))

            """
            NOTE:   Add additional parameters that are required by a `pymasq.metrics`
                    function or custom function, but cannot be set until runtime.
                    These parameters are limited to `orig_df`, `mod_df`, and `df`.
            """
            func_params = inspect.signature(func).parameters
            if "orig_df" in func_params:
                params["orig_df"] = self.target.copy()
            if "mod_df" in func_params:
                params["mod_df"] = target
            if "df" in func_params:
                params["df"] = target

            try:
                value = func(**params)
                assert isinstance(
                    value, (float, int)
                ), f"value ({value}) returned by {func.__name__} is not a float. It is {type(value)}"
            except Exception as e:
                if self.exit_on_error:
                    raise
                else:
                    if self.verbose >= 2:
                        print(f"[Warning] exception {func.__name__}: {e}")
                    raise
            fitnesses.append((func.__name__, value, args["weight"]))

        _, values, weights = zip(*fitnesses)
        wfit = np.average(values, weights=weights)
        return wfit, fitnesses

    def _safe_mutate(
        self, target: pd.DataFrame, mutations: Optional[List[Dict]] = None
    ) -> Union[pd.DataFrame, Dict[str, Dict], List[Exception]]:
        """Adds a try/except wrapper around _mutate"""
        try:
            new_target, mut_log = self._mutate(target, mutations)
        except Exception as e:
            # Mutation failed, so set new_target to target and record the exception in error_log
            if self.exit_on_error:
                raise e
            new_target = target
            error_log = [e]
            mut_log = {}
        else:
            error_log = []
        return new_target, mut_log, error_log

    @BEARTYPE
    def _mutate(
        self, target: pd.DataFrame, mutations: Optional[List[Dict]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        """Manipulate the `target` dataframe.

        Select and apply a mutation (e.g., mitigation) to alter the `target` dataframe.

        Parameters
        ----------
        target : pd.DataFrame
            The dataframe that will be manipulated.

        mutations : List[Dict], optional
            A list containing specific mitigation that can be applied at this mutation.
            If `None`, then all mitigations will be considered.

        Returns
        -------
        target : pd.DataFrame
            The altered, or mutated, `target` datafame. If `reuse_mutations` is set to `False`,
            and there are no mutations left, then the input `target` dataframe is returned as-is.

        Notes
        -----
        Mutations are selected at random if `randomize_mutations` is `True`.
        Otherwise`, they are selected in the order they appear in the `mutations` list.
        Setting `randomize_mutations` to `False` automatically sets `reuse_mutations`
        to `False`, therefore, mutations will be removed from consideration once they
        have been used.
        """
        # TODO: check if this prevents modification if solution not accepted
        target = target.copy()

        if not mutations:
            mutations = self._mutations

        if not self.reuse_mutations and not mutations:
            if self.verbose:
                print(
                    ">>> [NOOP] No mutations to apply (consider changing `reuse_mutations`)."
                )
            return target, {}  # NOOP; all mitigations used and removed

        mut = None
        if self.randomize_mutations:
            probs = [v["p"] for v in mutations]
            mut = np.random.choice(mutations, p=probs)
            if not self.reuse_mutations and mutations:
                # redistribute according to initial weighting
                mut_idx = mutations.index(mut)
                mutations.pop(mut_idx)
                probs.pop(mut_idx)
                prob_sum = sum(probs)
                # this condition is never violated; added for redundancy
                if prob_sum > 0:
                    for m in mutations:
                        m["p"] /= prob_sum
        else:
            if len(mutations) > 0:
                mut = mutations.pop(0)

        if mut is None:
            raise NoMutationAvailableError(mutations)

        func, args = [(k, v) for k, v in mut.items() if k != "p"][0]

        if isinstance(func, str):
            func = getattr(mits, func)

        if self.verbose >= 2:
            print("\t[Mutation]: %s" % (func), args)

        try:
            result = func(target, **args)
        except Exception as e:
            if self.verbose >= 2:
                print(f"[Warning] mutation {func.__name__} failed with args:={args}")
            raise
        if isinstance(result, pd.Series):
            col_args = args.get("col", args.get("cols", None))
            if col_args is None:
                raise ValueError(
                    "Invalid mitigation specification. `col` or `cols` is required. (Received: None)"
                )
            target[col_args] = result
        elif isinstance(result, pd.DataFrame):
            target[result.columns] = result
        else:
            raise ValueError(
                "Invalid object returned from mutation. Expected a Pandas DataFrame or Series. "
                f"(Received `{type(result).__name__}`)"
            )

        mut_log = {func.__name__: args}

        return target, mut_log


class Logbook:
    def __init__(self, headers: List[str]):
        """Logbook to hold statistics of each optimization run.

        Parameters
        ----------
        headers: List[str]
            Additional column headers specific to each optimization algorithm
            and user-specified evaluation `metrics`.
        """
        self.headers = self._pretty_headers(headers)
        self.log = pd.DataFrame(columns=["fitness", "mut_log"] + self.headers)

    @property
    def log(self) -> pd.DataFrame:
        """Get the `log` dataframe"""
        return self._log

    @log.setter
    def log(self, log: pd.DataFrame):
        """Set the log dataframe"""
        self._log = log

    @property
    def headers(self) -> List:
        """Get the columns of the `log` dataframe"""
        return self._headers

    @headers.setter
    def headers(self, headers: List[str]):
        """Set the headers dataframe"""
        self._headers = headers

    @BEARTYPE
    def _pretty_headers(self, headers: List) -> List[str]:
        """Return headers as list of strings.

        Parameters
        ----------
        headers : List of str or callable
            List of headers to use as column names in the `log` dataframe.

        Returns
        -------
        list
            String-formatted list of headers.
        """
        return [h if not callable(h) else h.__name__ for h in headers]

    @BEARTYPE
    def _pretty_values(self, record: Dict[str, Any]) -> List[Dict]:
        """Return formatted fitness values as headers.

        Parameters
        ----------
        record : Dict[str, Any]
            Key-values to append to the `log` dataframe.

        Returns
        -------
        List
            List containing the record Dict.
        """
        flog = record.pop("fit_log")
        for col, value, weight in flog:
            record[col] = (value, weight)
        return [record]

    @BEARTYPE
    def update(self, record: Dict[str, Any]):
        """Update the log with the parameter values.

        Parameters
        ----------
        record : Dict[str, Any]
            Key-values to append to the `log` dataframe.
        """
        record = self._pretty_values(record)
        df = pd.DataFrame.from_records(record)
        self.log = self.log.append(df, ignore_index=True)
