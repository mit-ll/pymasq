#!/usr/bin/env python
# coding: utf-8

from pymasq.datasets import load_census
from pymasq import metrics as mets
from pymasq.optimizations import apply_and_evaluate
from pymasq.optimizations import IterativeSearch


def test_apply_and_evaluate():
    # This checks that the output of apply_and_evaluate is the same as
    # if we called iterativeSearch(), which performs the metrics at each step

    df = load_census()[:100][["age", "education", "education_num"]]

    eval_fxns = {
        mets.k_anon: {"params": {"k": 2, "sensitive_col": "age"}},
        mets.propensity_score: {
            "params": {
                "sensitive_col": "age",
                "method": "larscv",
                "preprocessor": "label_encode",
            }
        },
    }
    mit_fxns = [
        {"hashing": {"cols": ["education"]}},
        {"top_recoding": {"cols": "age", "cutoff": 10, "to_val": 20}},
    ]
    mod, fit, log = apply_and_evaluate(df, mit_fxns, eval_fxns, verbose=0)

    both = IterativeSearch(
        iters=len(mit_fxns),
        target=df,
        mutations=mit_fxns,
        reuse_mutations=False,
        randomize_mutations=False,
        metrics=eval_fxns,
        verbose=0,
    )
    modb, fitb, logb = both.optimize()

    assert all(mod == modb)
    assert fit == fitb
    assert log.iloc[0].k_anon == logb.iloc[-1].k_anon
    assert log.iloc[0].propensity_score == logb.iloc[-1].propensity_score
