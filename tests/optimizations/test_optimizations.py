#!/usr/bin/env python
# coding: utf-8

import copy
import itertools
import json
import numpy as np
import pandas as pd
import pytest

from scipy.special import perm

import pymasq

pymasq.BEARTYPE = lambda func: func

from pymasq.datasets import load_census
from pymasq import optimizations as opts
from pymasq import mitigations as mits
from pymasq import set_seed

import random
from sklearn.utils import shuffle
import hashlib

set_seed(1)


@pytest.fixture
def my_df():
    df = load_census()
    cols = ["fnlwgt", "education", "marital_status", "sex", "capital_gain"]
    df = df.loc[:10, cols]
    return df


@pytest.fixture
def my_mutations():
    return [
        {
            mits.hashing: {
                "cols": "marital_status",
            },
            "p": 0.33,
        },
        {
            "add_noise": {
                "method": "additive",
                "cols": "fnlwgt",
                "magnitude": 2,
            },
            "p": 0.33,
        },
        {
            "topbot_recoding": {
                "cols": "capital_gain",
                "method": "both",
                "top_cutoff": 10,
                "top_to": 10,
                "bot_cutoff": 5,
                "bot_to": 4,
            },
            "p": 0.34,
        },
    ]


# evaluation functions
zeros = {lambda: 0: {"weight": 1}}
ones = {lambda: 1: {"weight": 1}}
rands = {lambda: np.random.rand(): {"weight": 1}}


# Test standard termination conditions
@pytest.mark.parametrize(
    "my_metrics, my_iters, my_theta",
    [
        (ones, 0, 0.0),  # iteration
        (ones, 2, 0.0),  # iteration
        (ones, np.inf, 1.0),  # theta
        (zeros, np.inf, 1.0),  # theta
        (zeros, np.inf, 0.0),  # theta
    ],
)
def test_optimizations_standard_termination(
    my_df, my_mutations, my_metrics, my_iters, my_theta
):
    """Test the standard termination condition of `pymasq.optimization` algorithms.

    The two termination conditions being checked are:
        1. Maximum no. of iterations reached
        2. Minimum theta reached

    Note
    ----
    `ExhuastiveSearch` is not part of this test since it's search procedure is guided
    primarily by additional parameters.
    """

    def _terminates_correctly(res, fit, log):
        assert any(
            [log.shape[0] == (my_iters + 1), log.iloc[-1]["fitness"] <= my_theta]
        )

    algo = opts.IterativeSearch(
        target=my_df,
        iters=my_iters,
        mutations=my_mutations,
        metrics=my_metrics,
        theta=my_theta,
        exit_on_error=True,
    )

    _terminates_correctly(*algo.optimize())

    algo = opts.IncrementalSearch(
        target=my_df,
        iters=my_iters,
        mutations=my_mutations,
        metrics=my_metrics,
        theta=my_theta,
        retry=np.inf,
        exit_on_error=True,
    )

    _terminates_correctly(*algo.optimize())

    algo = opts.StochasticSearch(
        target=my_df,
        iters=my_iters,
        mutations=my_mutations,
        metrics=my_metrics,
        theta=my_theta,
        exit_on_error=True,
    )

    _terminates_correctly(*algo.optimize())


# Test return values of optimizations
@pytest.mark.parametrize(
    "my_metrics, my_iters, my_theta",
    [
        (ones, 0, 0.0),
        (ones, 10, 0.0),
        (rands, 10, 0.0),
        (ones, np.inf, 1.0),
        (zeros, np.inf, 1.0),
        (zeros, np.inf, 0.0),
    ],
)
def test_optimizations_returns(my_df, my_mutations, my_metrics, my_iters, my_theta):
    """ Test the return variables of all `pymasq.optimization` algorithms. """

    def _returns_correctly(algo):
        result = algo.optimize()
        assert len(result) == 3
        res, fit, log = result
        assert isinstance(res, pd.DataFrame)
        assert res.shape == my_df.shape
        assert isinstance(fit, float)
        assert 0 <= fit <= 1.0
        assert isinstance(log, pd.DataFrame)
        assert any(
            [
                pytest.approx(log.loc[:, "fitness"].min(), 0.00001) == fit,
                pytest.approx(log.iloc[-1]["fitness"], 0.00001) == fit,
            ]
        )

    algo = opts.IterativeSearch(
        target=my_df,
        iters=my_iters,
        mutations=my_mutations,
        metrics=my_metrics,
        theta=my_theta,
        exit_on_error=True,
    )

    _returns_correctly(algo)

    algo = opts.IncrementalSearch(
        target=my_df,
        iters=my_iters,
        mutations=my_mutations,
        metrics=my_metrics,
        theta=my_theta,
        retry=np.inf,
        exit_on_error=True,
    )

    _returns_correctly(algo)

    algo = opts.StochasticSearch(
        target=my_df,
        iters=my_iters,
        mutations=my_mutations,
        metrics=my_metrics,
        theta=my_theta,
        exit_on_error=True,
    )

    _returns_correctly(algo)

    algo = opts.ExhaustiveSearch(
        target=my_df,
        iters=my_iters,
        mutations=my_mutations,
        metrics=my_metrics,
        theta=my_theta,
        exit_on_error=True,
    )

    _returns_correctly(algo)


# Test reuse mutations
@pytest.mark.parametrize(
    "my_metrics, my_iters, my_reuse_mutations", [(ones, 10, True), (ones, 10, False)]
)
def test_reuse_mutations(my_df, my_mutations, my_metrics, my_iters, my_reuse_mutations):
    """
    Tests the `reuse_mutation` flag used by `pymasq.optimizations`.
    """

    def _reuse_mutations_correctly(res, fit, log):
        mut_log = log.loc[1:, ["mut_log"]].values.flatten()
        mut_log_unique = list(map(json.loads, set(map(json.dumps, mut_log))))

        _my_mutations = [
            dict((k if isinstance(k, str) else k.__name__, v) for k, v in m.items())
            for m in my_mutations
        ]
        [_.pop("p") for _ in _my_mutations]

        assert any(
            [
                all([m in _my_mutations for m in mut_log]),
                len(mut_log_unique) == len(my_mutations) + 1,
            ]
        )

    algo = opts.IterativeSearch(
        target=my_df,
        iters=my_iters,
        mutations=copy.deepcopy(my_mutations),
        metrics=my_metrics,
        theta=0.0,
        reuse_mutations=my_reuse_mutations,
        exit_on_error=True,
    )

    _reuse_mutations_correctly(*algo.optimize())

    algo = opts.IncrementalSearch(
        target=my_df,
        iters=my_iters,
        mutations=copy.deepcopy(my_mutations),
        metrics=my_metrics,
        theta=0.0,
        retry=np.inf,
        reuse_mutations=my_reuse_mutations,
        exit_on_error=True,
    )

    _reuse_mutations_correctly(*algo.optimize())

    algo = opts.StochasticSearch(
        target=my_df,
        iters=my_iters,
        mutations=copy.deepcopy(my_mutations),
        metrics=my_metrics,
        theta=0.0,
        reuse_mutations=my_reuse_mutations,
        exit_on_error=True,
    )

    _reuse_mutations_correctly(*algo.optimize())


# Test randomize_mutations
@pytest.mark.parametrize(
    "my_metrics, my_iters, my_randomize_mutations",
    [(ones, 10, True), (ones, 10, False)],
)
def test_randomize_mutations(
    my_df, my_mutations, my_metrics, my_iters, my_randomize_mutations
):
    """
    Tests the `randomize_mutation` flag used by `pymasq.optimizations`.

    Note
    ----
    When `True`, `randomize_mutation` will force mutations to be chosen at random,
    rather than in the order they were specified. Though unlikely, the ordering of
    mutations chosen at random may be the exact same order they were specified.
    To account for this case, this test runs 50 tests (`N_TESTS` = 50) with a
    required success rate of 80% (`PASS_RATE` = 0.80).
    """

    def _randomize_mutations_correctly(res, fit, log):
        mut_log = log.loc[1:, ["mut_log"]].values.flatten()
        mut_log_unique = list(map(json.loads, set(map(json.dumps, mut_log))))

        _my_mutations = [
            dict((k if isinstance(k, str) else k.__name__, v) for k, v in m.items())
            for m in my_mutations
        ]
        [_.pop("p") for _ in _my_mutations]

        return any(
            [
                len(mut_log_unique) == len(my_mutations),  # randomize = True
                all(
                    mut_log[: len(_my_mutations)] == _my_mutations
                ),  # randomize = False
            ]
        )

    N_TESTS = 50
    PASS_RATE = 0.80

    result = []
    for _ in range(N_TESTS):
        algo = opts.IterativeSearch(
            target=my_df,
            iters=my_iters,
            mutations=copy.deepcopy(my_mutations),
            metrics=my_metrics,
            theta=0.0,
            randomize_mutations=my_randomize_mutations,
            exit_on_error=True,
        )
        result.append(1 if _randomize_mutations_correctly(*algo.optimize()) else 0)

    assert sum(result) / N_TESTS >= PASS_RATE

    result = []
    for _ in range(N_TESTS):
        algo = opts.IncrementalSearch(
            target=my_df,
            iters=my_iters,
            mutations=copy.deepcopy(my_mutations),
            metrics=my_metrics,
            theta=0.0,
            retry=np.inf,
            randomize_mutations=my_randomize_mutations,
            exit_on_error=True,
        )
        result.append(1 if _randomize_mutations_correctly(*algo.optimize()) else 0)

    assert sum(result) / N_TESTS >= PASS_RATE

    result = []
    for _ in range(N_TESTS):
        algo = opts.StochasticSearch(
            target=my_df,
            iters=my_iters,
            mutations=copy.deepcopy(my_mutations),
            metrics=my_metrics,
            theta=0.0,
            randomize_mutations=my_randomize_mutations,
            exit_on_error=True,
        )
        result.append(1 if _randomize_mutations_correctly(*algo.optimize()) else 0)

    assert sum(result) / N_TESTS >= PASS_RATE


# Test IncrementalSearch
@pytest.mark.parametrize(
    "my_metrics, my_iters, my_theta, my_retry",
    [
        (ones, np.inf, 0.0, 1),
        (ones, np.inf, 0.0, 100),
    ],
)
def test_IncrementalSearch(
    my_df, my_mutations, my_metrics, my_iters, my_theta, my_retry
):
    """
    Tests the `optimization.IncrementalSearch`'s termination condition based on `retry`.
    """
    algo = opts.IncrementalSearch(
        target=my_df,
        iters=my_iters,
        mutations=my_mutations,
        metrics=my_metrics,
        theta=my_theta,
        retry=my_retry,
        exit_on_error=True,
    )

    _, _, log = algo.optimize()

    assert log.iloc[0]["retry"] == my_retry
    assert log.iloc[-1]["retry"] == 0
    assert log.shape[0] == my_retry + 1


# Test ExhaustiveSearch
@pytest.mark.parametrize(
    "my_metrics, my_iters, my_theta, my_num_perms, my_size_perms",
    [
        (ones, np.inf, 0.0, 1, 1),  # permutations
        (ones, np.inf, 0.0, 1, None),  # permutations
        (ones, np.inf, 0.0, None, 1),  # permutations
        (ones, np.inf, 0.0, None, None),  # permutations
        (ones, 0, 0.0, None, None),  # iterations
        (ones, 1, 0.0, None, None),  # iterations
        (zeros, np.inf, 1.0, None, None),  # theta
    ],
)
def test_ExhaustiveSearch(
    my_df, my_mutations, my_metrics, my_iters, my_theta, my_num_perms, my_size_perms
):
    """
    Tests the `optimization.Exhaustive`'s termination condition based on `iters`,
    `theta`, `num_perms` and `size_perms`.
    """

    def _terminates_correctly(res, fit, log):
        if not np.isinf(my_iters):
            assert log.shape[0] == (my_iters + 1)
        elif my_theta == 1.0 or my_theta == 0.9:
            assert log.iloc[-1]["fitness"] <= my_theta
        else:
            # terminates via permutations
            _my_size_perms = (
                len(my_mutations)
                if not my_size_perms
                else min(len(my_mutations), my_size_perms)
            )
            _max_perms = perm(len(my_mutations), _my_size_perms)
            _my_num_perms = (
                _max_perms if not my_num_perms else min(my_num_perms, _max_perms)
            )

            perms = list(
                itertools.permutations(range(len(my_mutations)), _my_size_perms)
            )

            assert log.iloc[-1]["perm_num"] == _my_num_perms
            assert log[1:].shape[0] == _my_num_perms * _my_size_perms

    algo = opts.ExhaustiveSearch(
        target=my_df,
        iters=my_iters,
        mutations=my_mutations,
        metrics=my_metrics,
        theta=my_theta,
        num_perms=my_num_perms,
        size_perms=my_size_perms,
        exit_on_error=True,
    )

    _terminates_correctly(*algo.optimize())


def test_exit_on_error():
    def throw_error_mut(*args, **kwargs):
        df = args[0]
        choice = random.choices([True, False], weights=[1, 2])
        if choice[0] == True:
            raise Exception("Mutation error thrown on purpose.")
        else:
            df = shuffle(df)
            df.reset_index(inplace=True, drop=True)
            return df

    def rand_metric(df, *args, **kwargs):
        Hash = hashlib.sha512
        MAX_HASH_PLUS_ONE = 2 ** (Hash().digest_size * 8)
        seed = str(df).encode()
        hash_digest = Hash(seed).digest()
        hash_int = int.from_bytes(hash_digest, "big")
        return np.round(hash_int / MAX_HASH_PLUS_ONE, 4)  # Float division

    def throw_error_metric(df, *args, **kwargs):
        choice = random.choices([True, False], weights=[1, 2])
        if choice[0] == True:
            raise Exception("Metrics error thrown on purpose.")
        else:
            return rand_metric(df)

    my_err_mutations = [
        {
            throw_error_mut: {"mutatation": 1, "df": True},
        },
        {
            throw_error_mut: {"mutation": 2, "df": True},
        },
    ]

    error_metrics = {
        throw_error_metric: {},
    }

    def _terminates_correctly(res, fit, log):
        # number of errors in each iteration
        log["num_error"] = log.apply(lambda row: len(row.error_log), axis=1)
        # if fitness is np.inf, there there should be at least one error
        # if there are not errors, then fitness should be between 0 and 1
        log["test"] = log.apply(
            lambda row: ((row.fitness == np.inf) and (row.num_error >= 0))
            or ((row.fitness >= 0.0) and (row.fitness <= 1.0)),
            axis=1,
        )
        # all of the above are true
        assert all(log.test)
        # it shouldn't be that every row was an error
        assert (
            min(log.num_error) == 0
        ), f"it shouldn't be that every row was an error. Min(log.num_error)={min(log.num_error)}"
        # it shouldn't be that we had no errors
        assert max(log.num_error) > 0, "it shouldn't be that we had no errors."
        # make sure we didn't stop after only 5 iterations for some unexplained reason
        assert len(log) > 5, "Very few iterations. Did this exit early?"
        # all errors should be of the type we created
        log["test2"] = log.error_log.apply(
            lambda row: all(
                [
                    str(x)
                    in [
                        "Metrics error thrown on purpose.",
                        "Mutation error thrown on purpose.",
                    ]
                    for x in row
                ]
            )
        )
        assert all(log.test2)

    algo = opts.IterativeSearch(
        target=pd.DataFrame(
            data={"col1": [1, 2, 3, 4, 5, 6], "col2": [30, 40, 50, 60, 70, 80]}
        ),
        iters=1000,
        mutations=my_err_mutations,
        metrics=error_metrics,
        theta=0.0,
        verbose=0,
        reuse_mutations=True,
        exit_on_error=False,
    )
    _terminates_correctly(*algo.optimize())

    algo = opts.IncrementalSearch(
        target=pd.DataFrame(
            data={"col1": [1, 2, 3, 4, 5, 6], "col2": [30, 40, 50, 60, 70, 80]}
        ),
        iters=1000,
        mutations=my_err_mutations,
        metrics=error_metrics,
        theta=0.0,
        verbose=0,
        retry=1000,
        reuse_mutations=True,
        exit_on_error=False,
    )
    _terminates_correctly(*algo.optimize())

    algo = opts.StochasticSearch(
        target=pd.DataFrame(
            data={"col1": [1, 2, 3, 4, 5, 6], "col2": [30, 40, 50, 60, 70, 80]}
        ),
        iters=1000,
        mutations=my_err_mutations,
        metrics=error_metrics,
        theta=0.0,
        verbose=0,
        reuse_mutations=True,
        exit_on_error=False,
    )
    _terminates_correctly(*algo.optimize())

    algo = opts.ExhaustiveSearch(
        target=pd.DataFrame(
            data={"col1": [1, 2, 3, 4, 5, 6], "col2": [30, 40, 50, 60, 70, 80]}
        ),
        iters=1000,
        mutations=my_err_mutations
        + my_err_mutations,  # so that we have more iterations
        metrics=error_metrics,
        theta=0.0,
        verbose=0,
        reuse_mutations=True,
        exit_on_error=False,
    )
    _terminates_correctly(*algo.optimize())
