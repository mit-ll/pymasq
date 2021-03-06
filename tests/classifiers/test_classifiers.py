#!/usr/bin/env python
# coding: utf-8

import shutil
import pytest
import pymasq.config as cfg
from pathlib import Path
from pymasq.datasets import load_census
from pymasq.preprocessing import LabelEncoder_pm, EmbeddingsEncoder
from pymasq.models.models import (
    LogisticRegressionClassifier,
    TpotClassifier,
    RFClassifier,
)


@pytest.fixture
def my_df():
    df = load_census()
    cols = [
        "age",
        "fnlwgt",
        "education",
        "marital_status",
        "sex",
        "capital_gain",
        "income_level",
    ]
    df = df.loc[:1000, cols]
    return df


@pytest.mark.parametrize(
    "combo",
    [
        (LogisticRegressionClassifier, LabelEncoder_pm, 0.5),
        (LogisticRegressionClassifier, EmbeddingsEncoder, 0.5),
        (RFClassifier, LabelEncoder_pm, 1.0),
        (RFClassifier, EmbeddingsEncoder, 1.0),
        (TpotClassifier, LabelEncoder_pm, 0.77),
        (TpotClassifier, EmbeddingsEncoder, 0.86),
    ],
)
def test_classifiers(my_df, combo):
    classifier_type, preprocessor, answer = combo
    print(classifier_type)
    # check that the classifier gets the expected value given a set hmac key and set seed

    dir_name = "cache_test"
    Path(dir_name).mkdir(exist_ok=True)

    cfg.CACHE_HMAC_KEY = "my key"
    classifier = classifier_type(cache_location=dir_name, modeling_task="binary")

    # seed is an argument for embeddings only
    if preprocessor == EmbeddingsEncoder:
        classifier.train(
            df=my_df,
            y_column="sex",
            preprocessor=preprocessor,
            verbose=1,
            retrain=True,
        )
    else:
        classifier.train(
            df=my_df,
            y_column="sex",
            preprocessor=preprocessor,
            verbose=1,
            retrain=True,
        )
    # should make use of cache
    enc = preprocessor.encode(my_df, cache_location=dir_name, verbose=1)

    print(type(enc.drop(["sex"], axis=1)))
    print(type(enc.sex))
    score = classifier.predict(x_test=enc.drop(["sex"], axis=1), y_true=enc.sex)
    print(f"{classifier.name}, {preprocessor}: {score}")
    assert round(score, 2) == answer, "Scores should match (trial {}, {})".format(
        classifier_type, preprocessor
    )

    # Check if the cached file loads, and that the hmac checks out
    print(f"\n{classifier.name}, {preprocessor} load")
    classifier.load_trained_model(my_df, verbose=1)

    print("removing cache")
    shutil.rmtree(dir_name)
