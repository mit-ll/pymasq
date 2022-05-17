import itertools
from typing import Optional

import numpy as np
from scipy.special import perm

from pymasq import BEARTYPE
from pymasq.errors import LessThanOrEqualToZeroError, NotInRangeError
from pymasq.optimizations._base import OptimizationBase


class IterativeSearch(OptimizationBase):
    """Iterative (sequential) optimization algorithm.

    Iterative search consists solely of applying each of the selected mitigations
    on a target dataframe. This search procedure is only concerned with exploration,
    and not exploitation, since it does not keep track of the best mitigation strategy so far.

    Parameters
    ----------
    return_best : bool, optional (Default: False)
        Return the best target and fitness value found or the last one found.

    Notes
    -----
    The iterative method is a mathematical procedure that uses an initial guess to generate a
    sequence of improving approximate solutions for a class of problems, in which the n-th
    approximation is derived from the previous ones. A specific implementation of an iterative method,
    including the termination criteria, is an algorithm of the iterative method. [1]_

    If self.exit_on_error==False, then errors are written to the error_log column
    of the returned logbook. Each row contains a list of errors.
    Each item in a row's list is of type exception.
    An empty list indicates there were no errors.

    References
    ----------
    https://en.wikipedia.org/wiki/Iterative_method

    Examples
    --------
    Import the dataset, evaluation metrics, mitigations, and optimization procedure

    >>> from pymasq.datasets import load_census
    >>> from pymasq.optimizations import IterativeSearch
    >>> from pymasq.metrics import k_anon
    >>> from pymasq.mitigations import add_noise

    Load the census dataset

    >>> df = load_census()

    Set the evaluation function to be the k-anonymity risk metric and set
    its respective parameters.

    >>> eval_fxns = {
           k_anon: {"weight": 1., "params": {"key_vars": ["sex"]}}
       }

    Set the mitigation functions to be used during the optimization procedure and
    set its respective parameters.

    >>> mit_fxns = [
           {"p": 1., add_noise: {"col": ["capital_gain"]}}
       ]

    Run the optimization procedure for 10 iterations.

    >>> search = IterativeSearch(
           target=df, iters=10, mutations=mit_fxns, metrics=eval_fxns
       )
    >>> result_df, fit, logbook = search.optimize()
    >>> result_df

    """

    @BEARTYPE
    def __init__(self, *args, return_best: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_best = return_best

    @BEARTYPE
    def _optimize(self):
        target = self._target
        # Initial evaluation
        cur_fit, fit_log, error_log = self._safe_evaluate(target)
        self._record_stats(
            fitness=cur_fit, mut_log={}, fit_log=fit_log, error_log=error_log
        )

        best_target = target
        best_fit = cur_fit

        while all([cur_fit > self.theta, self._iters > 0]):

            if self.verbose:
                print("-- Iteration [%i] --" % (self._max_iters - self._iters))
                if self.progress_reporter:
                    self.progress_reporter(
                        round(1 - (self._iters / (self._max_iters * 1.0)), 2)
                    )

            # New target from mutation
            new_target, mut_log, error_log = self._safe_mutate(target)
            # Evaluation
            new_fit, fit_log, met_errors = self._safe_evaluate(new_target)
            error_log += met_errors
            if self.verbose >= 2:
                print(
                    ">> Current fitness: %.5f | " % (cur_fit),
                    "New fitness: %.5f | " % (new_fit),
                    "Best fitness: %.5f" % (best_fit),
                )
                if self.verbose >= 3:
                    print(new_target)

            cur_fit = new_fit
            target = new_target

            if new_fit <= best_fit:
                best_target = new_target
                best_fit = new_fit

            self._record_stats(
                fitness=cur_fit, mut_log=mut_log, fit_log=fit_log, error_log=error_log
            )

            if cur_fit <= self.theta and self.verbose:
                print(">>> [Terminating]: Solution found")

            self._iters -= 1

            if self._iters <= 0 and self.verbose:
                print(">>> [Terminating]: Iterations complete")

        if self.return_best:
            return best_target, best_fit, self._logbook

        return target, cur_fit, self._logbook


class IncrementalSearch(OptimizationBase):
    """Hill climbing optimization algorithm.

    Incremental search applies a mitigation strategy on a target dataframe,
    keeping only the mitigations which resulted in an improved fitness value and discarding all the rest.
    Thus, this search procedure is primarily focused on exploitation, rather than exploration,
    since it only accepts solutions that imrpove its fitness value. The search procedure is
    terminated upon encountering a mitigation of that worsens its fitness value. A parameter,
    `retry`, can be specified to allow the search procedure to continue even if it encounters a mitigation
    which results in a worse fitness value.

    Parameters
    ----------
    retry : int, optional (Default: 3)
        Number of times to continue an optimization procedure
        even if a worse solution is found. Note that the standard hill-climbing
        algorithm terminates as soon as one inferior solution is reached (e.g.
        `retry` is `0).

    Notes
    -----
    Hill climbing is a mathematical optimization technique which belongs to the family of local search.
    It is an iterative algorithm that starts with an arbitrary solution to a problem, then attempts to
    find a better solution by making an incremental change to the solution. If the change produces a
    better solution, another incremental change is made to the new solution, and so on until no further
    improvements can be found. [1]_

    If self.exit_on_error==False, then errors are written to the error_log column
    of the returned logbook. Each row contains a list of errors.
    Each item in a row's list is of type exception.
    An empty list indicates there were no errors.

    References
    ----------
    https://en.wikipedia.org/wiki/Hill_climbing

    Examples
    --------
    Import the dataset, evaluation metrics, mitigations, and optimization procedure

    >> from pymasq.datasets import load_census
    >> from pymasq.optimizations import IncrementalSearch
    >> from pymasq.metrics import k_anon
    >> from pymasq.mitigations import add_noise

    Load the census dataset

    >> df = load_census()

    Set the evaluation function to be the k-anonymity risk metric and set
    its respective parameters.

    >> eval_fxns = {
           k_anon: {"weight": 1., "params": {"key_vars": ["sex"]}}
       }

    Set the mitigation functions to be used during the optimization procedure and
    set its respective parameters.

    >> mit_fxns = [
           {"p": 1., add_noise: {"col": ["capital_gain"]}}
       ]

    Run the optimization procedure for 10 iterations with 5 retries.

    >> search = IncrementalSearch(
           target=df, iters=10, mutations=mit_fxns, metrics=eval_fxns, retry=5
       )
    >> result_df, fit, logbook = search.optimize()
    >> result_df

    """

    @BEARTYPE
    def __init__(self, *args, retry: int = 3, **kwargs):
        kwargs["headers"] = ["retry"]
        super().__init__(*args, **kwargs)

        self.retry = retry
        if self.retry <= 0:
            raise LessThanOrEqualToZeroError(
                f"Number of retries (`retry`) must be > 0. (Received: {retry})"
            )

    @BEARTYPE
    def _optimize(self):

        target = self._target
        retry = self.retry

        # Initial Evaluation
        cur_fit, fit_log, error_log = self._safe_evaluate(target)
        self._record_stats(
            fitness=cur_fit,
            mut_log={},
            fit_log=fit_log,
            retry=retry,
            error_log=error_log,
        )

        while all([cur_fit > self.theta, self._iters > 0, retry > 0]):

            if self.verbose:
                print("-- Iteration [%i] --" % (self._max_iters - self._iters))
                if self.progress_reporter:
                    self.progress_reporter(
                        round(1 - (self._iters / (self._max_iters * 1.0)), 2)
                    )

            # New target from mutation
            new_target, mut_log, error_log = self._safe_mutate(target)
            # Evaluation
            new_fit, fit_log, met_errors = self._safe_evaluate(new_target)
            error_log += met_errors

            if self.verbose >= 2:
                print(
                    ">> Current fitness: %.5f | New fitness: %.5f" % (cur_fit, new_fit)
                )
                if self.verbose >= 3:
                    print(new_target)

            if new_fit < cur_fit:
                cur_fit = new_fit
                target = new_target
            else:
                retry -= 1
                if self.verbose >= 2:
                    print(">>> Retries left: %i" % (retry))

            self._record_stats(
                fitness=cur_fit,
                mut_log=mut_log,
                fit_log=fit_log,
                retry=retry,
                error_log=error_log,
            )

            if cur_fit <= self.theta and self.verbose:
                print(">>> [Terminating]: Solution found")

            if retry <= 0:
                if self.verbose:
                    print(">>> [Terminating]: Max number of retries reached")

            self._iters -= 1

            if self._iters <= 0 and self.verbose:
                print(">>> [Terminating]: Iterations complete")

        return target, cur_fit, self._logbook


class StochasticSearch(OptimizationBase):
    """Standard simulated annealing optimization algorithm.

    Stochastic search balances exploration and exploitation, by applying a mitigation strategy
    on a target dataframe and accepting both inferior and superior solutions. This procedure
    keeps track of the overall best solution found, while also improving upon a current target dataframe.
    This allows the search procedure to explore and evaluate sufficient candidate solutions, and
    ensures the best mitigation strategy is found for the parameters specified. Two parameters,
    `alpha` and `temperature`, specify the amount of exploration to perform. A high `temperature`
    encourages the procedure to accept solutions that don't improve the current fitness value for the
    sake of exploration. The `temperature` is reduced over each iteration, following a standard
    annealing schedule, parameterized by `alpha`. Once a low `temperature` is reached, the procedure
    will only accept solutions that improved the current fitness value.

    Parameters
    ----------
    temperature : float, optional (Default: 1.0)
        Starting value for accepting inferior solutions. This value will decay
        using the standard annealing schedule below.

        .. math::

            temperature := temperature x (1 - `alpha`)

    alpha : float, optional (Default: 0.05)
        The `temperature` annealing/decay rate. High exploration is achieved when
        `alpha` is low (near zero) and low exploration when `alpha is high (near one).

    Notes
    -----
    Simulated annealing (SA) is a probabilistic technique for approximating the global optimum of a given function.
    Specifically, it is a metaheuristic to approximate global optimization in a large search space for an optimization problem.
    It is often used when the search space is discrete (e.g., the traveling salesman problem). For problems where finding an
    approximate global optimum is more important than finding a precise local optimum in a fixed amount of time,
    simulated annealing may be preferable to alternatives such as gradient descent. [1_]

    If self.exit_on_error==False, then errors are written to the error_log column
    of the returned logbook. Each row contains a list of errors.
    Each item in a row's list is of type exception.
    An empty list indicates there were no errors.

    References
    ----------
    https://en.wikipedia.org/wiki/Simulated_annealing

    Examples
    --------
    Import the dataset, evaluation metrics, mitigations, and optimization procedure

    >> from pymasq.datasets import load_census
    >> from pymasq.optimizations import StochasticSearch
    >> from pymasq.metrics import k_anon
    >> from pymasq.mitigations import add_noise

    Load the census dataset

    >> df = load_census()

    Set the evaluation function to be the k-anonymity risk metric and set
    its respective parameters.

    >> eval_fxns = {
           k_anon: {"weight": 1., "params": {"key_vars": ["sex"]}}
       }

    Set the mitigation functions to be used during the optimization procedure and
    set its respective parameters.

    >> mit_fxns = [
           {"p": 1., add_noise: {"col": ["capital_gain"]}}
       ]

    Run the optimization procedure for 10 iterations with an `alpha` decay of 0.25.

    >> search = StochasticSearch(
           target=df, iters=10, mutations=mit_fxns, metrics=eval_fxns, alpha=0.25
       )
    >> result_df, fit, logbook = search.optimize()
    >> result_df

    """

    @BEARTYPE
    def __init__(self, *args, alpha: float = 0.05, temperature: float = 1.0, **kwargs):
        kwargs["headers"] = ["accepted"]
        super().__init__(*args, **kwargs)

        self.temperature = temperature
        self.alpha = alpha
        if not 0.0 <= temperature <= 1.0:
            raise NotInRangeError(
                f"Temperature must be between interval [0,1], inclusive. (Received: {temperature})"
            )
        if not 0.0 <= alpha <= 1.0:
            raise NotInRangeError(
                f"Alpha must be between interval [0,1], inclusive. (Received: {alpha})"
            )

    @BEARTYPE
    def _accept_prob(self, cur_fit: float, new_fit: float) -> float:
        """Calculate the acceptance probability of an inferior solution

        Parameters
        ----------
        cur_fit : float
            Fitness value of the current solution

        new_fit : float
            Fitness value of the new solution

        Returns
        -------
        float
            Probability of acceptance.
        """
        if new_fit <= cur_fit:
            return 1.0
        return np.exp((cur_fit - new_fit) / self.temperature)

    @BEARTYPE
    def _optimize(self):
        target = self._target

        # Initial evaluation
        cur_fit, fit_log, error_log = self._safe_evaluate(target)
        self._record_stats(
            fitness=cur_fit,
            mut_log={},
            fit_log=fit_log,
            accepted=True,
            error_log=error_log,
        )

        best_target = target
        best_fit = cur_fit

        while all([best_fit > self.theta, self._iters > 0]):

            if self.verbose:
                print("-- Iteration [%i] --" % (self._max_iters - self._iters))
                if self.progress_reporter:
                    self.progress_reporter(
                        round(1 - (self._iters / (self._max_iters * 1.0)), 2)
                    )

            accepted = False

            # New target from mutation
            new_target, mut_log, error_log = self._safe_mutate(target)
            # Evaluation
            new_fit, fit_log, met_errors = self._safe_evaluate(new_target)
            error_log += met_errors

            if self.verbose >= 2:
                print(
                    ">> Current fitness: %.5f | " % (cur_fit),
                    "New fitness: %.5f | " % (new_fit),
                    "Best fitness: %.5f" % (best_fit),
                )
                if self.verbose >= 3:
                    print(new_target)

            prob = np.random.random_sample()

            if (target.equals(new_target) == False) and (
                self._accept_prob(cur_fit, new_fit) > prob
            ):
                if self.verbose >= 1:
                    print(
                        ">> New solution accepted",
                        "(inferior solution)" if cur_fit < new_fit else "",
                    )
                cur_fit = new_fit
                target = new_target
                accepted = True

            if new_fit < best_fit:
                if self.verbose >= 1:
                    print(f">> New [best] solution found: {new_fit} < {best_fit}")
                best_fit = new_fit
                best_target = new_target

            self._record_stats(
                fitness=cur_fit,
                mut_log=mut_log,
                fit_log=fit_log,
                accepted=accepted,
                error_log=error_log,
            )

            self._iters -= 1
            self.temperature *= 1 - self.alpha

            if cur_fit <= self.theta and self.verbose:
                print(">>> [Terminating]: Solution found")

            if self._iters <= 0 and self.verbose:
                print(">>> [Terminating]: Iterations complete")

        return best_target, best_fit, self._logbook


class ExhaustiveSearch(OptimizationBase):
    """Standard brute-force optimization algorithm.

    Exhuastive search is a pure, brute-force approach at finding an optimal mitigation strategy
    based on the specified mitigations and evaluation metrics. This procedure will apply and evaluation
    each permutation of the specified mitigations, thus guaranting to find the best, or optimal,
    mitigation strategy. Exploring and evaluating each mitigation permutation can be computationally
    expensive. The parameters, `num_perms` and `size_perms` can be specified to limit the number of
    permutations explored and the length of each permutation, respectively. The search procedure is
    terminated once `num_perms` is reached. Note that this restricts the exploration process in which
    finding the optimial mitigation strategy can no longer be guaranteed.

    Parameters
    ----------
    num_perms : int, optional
        Number of allowed permutations. If not set, all permutations up to
        `size_perms` will be enumerated.
        Note that this is may be computationally expensive and time consuming.

    size_perms : int, optional
        The maximum size of each permutation. If not set, all permutations will be
        enumerated. Note that this is may be computationally expensive and time consuming.

    return_best : bool, optional (Default: False)
        Return the best target and fitness value found or the last one found.

    Notes
    -----
    Brute-force search or exhaustive search, also known as generate and test, is a very general
    problem-solving technique and algorithmic paradigm that consists of systematically enumerating
    all possible candidates for the solution and checking whether each candidate satisfies the
    problem's statement. [1]_

    If self.exit_on_error==False, then errors are written to the error_log column
    of the returned logbook. Each row contains a list of errors.
    Each item in a row's list is of type exception.
    An empty list indicates there were no errors.

    References
    ----------
    https://en.wikipedia.org/wiki/Brute-force_search

    Examples
    --------
    Import the dataset, evaluation metrics, mitigations, and optimization procedure

    >> from pymasq.datasets import load_census
    >> from pymasq.optimizations import ExhaustiveSearch
    >> from pymasq.metrics import k_anon
    >> from pymasq.mitigations import add_noise, map_values, truncate_values

    Load the census dataset

    >> df = load_census()

    Set the evaluation function to be the k-anonymity risk metric and set
    its respective parameters.

    >> eval_fxns = {
           k_anon: {"weight": 1., "params": {"key_vars": ["sex"]}}
       }

    Set the mitigation functions to be used during the optimization procedure and
    set its respective parameters.

    >> mit_fxns = [
           {"p": 0.33, add_noise: {"col": ["capital_gain"]}},
           {"p": 0.33, map_values: {"col": ["education"], "method": "md5"}},
           {"p": 0.34, truncate_values: {"col": ["education"], "end_idx": 3}}
       ]

    Run the optimization procedure for 10 iterations with the size of each
    permutation set to 2 and the number of allowed permutations to 4 (out of 6 possible).

    >> search = ExhaustiveSearch(
           target=df, iters=10, mutations=mit_fxns, metrics=eval_fxns, size_perms=2, num_perms=4
       )
    >> result_df, fit, logbook = search.optimize()
    >> result_df
    """

    @BEARTYPE
    def __init__(
        self,
        *args,
        num_perms: Optional[int] = None,
        size_perms: Optional[int] = None,
        return_best: bool = False,
        **kwargs,
    ):

        kwargs["headers"] = ["perm_num"]
        super().__init__(*args, **kwargs)

        self.return_best = return_best

        self.size_perms = (
            len(self.mutations)
            if not size_perms
            else min(len(self.mutations), size_perms)
        )
        if self.size_perms <= 0:
            raise LessThanOrEqualToZeroError(
                f"Size of each permutation (`size_perms`) must be > 0. (Received: {self.size_perms})"
            )

        max_perms = perm(len(self.mutations), self.size_perms)
        self.max_perms = max_perms if not num_perms else min(max_perms, num_perms)
        if self.max_perms <= 0:
            raise LessThanOrEqualToZeroError(
                f"Number of permutations (`num_perms`) must be > 0. (Received: {self.max_perms})"
            )

    @BEARTYPE
    def _optimize(self):
        target = self._target.copy()

        # Initial evaluation
        cur_fit, fit_log, error_log = self._safe_evaluate(target)
        self._record_stats(
            fitness=cur_fit,
            mut_log={},
            fit_log=fit_log,
            perm_num=0,
            error_log=error_log,
        )

        best_target = target
        best_fit = cur_fit

        if any([cur_fit <= self.theta, self._iters <= 0]):
            if self.verbose:
                print(">>> [Terminating]: Solution found or Iterations Complete")
            return target, cur_fit, self._logbook

        if self.randomize_mutations:
            # Note: only matters when `num_perms` is set.
            test = np.random.shuffle(self._mutations)

        for num_perms, mutation_perms in enumerate(
            itertools.permutations(self._mutations, self.size_perms)
        ):
            if self.verbose:
                print("-- Permutation: [%i] --" % (num_perms))

            target = self._target.copy()

            stop = False
            for mutation in mutation_perms:

                if self.verbose:
                    print("\t-- Iteration [%i] --" % (self._max_iters - self._iters))
                    if self.progress_reporter:
                        self.progress_reporter(
                            round(1 - (self._iters / (self._max_iters * 1.0)), 2)
                        )

                mutation["p"] = 1.0

                # New target from mutation
                new_target, mut_log, error_log = self._safe_mutate(target, [mutation])
                # Evaluation
                new_fit, fit_log, met_errors = self._safe_evaluate(new_target)
                error_log += met_errors

                if self.verbose >= 2:
                    print(
                        ">> Current fitness: %.5f | " % (cur_fit),
                        "New fitness: %.5f | " % (new_fit),
                        "Best fitness: %.5f" % (best_fit),
                    )
                    if self.verbose >= 3:
                        print(new_target)

                cur_fit = new_fit
                target = new_target

                if new_fit < best_fit:
                    best_target = new_target
                    best_fit = new_fit

                self._record_stats(
                    fitness=cur_fit,
                    mut_log=mut_log,
                    fit_log=fit_log,
                    perm_num=num_perms + 1,
                    error_log=error_log,
                )

                if cur_fit <= self.theta:
                    if self.verbose:
                        print(">>> [Terminating]: Solution found")
                    stop = True
                    break

                self._iters -= 1

                if self._iters <= 0:
                    if self.verbose:
                        print(">>> [Terminating]: Iterations complete")
                    stop = True
                    break

            if self.verbose:
                print("\n")

            if (num_perms + 1) >= self.max_perms:
                if self.verbose:
                    print(">>> [Terminating]: Number of permutations complete")
                stop = True

            if stop:
                break

        if self.return_best:
            return best_target, best_fit, self._logbook

        return target, cur_fit, self._logbook
