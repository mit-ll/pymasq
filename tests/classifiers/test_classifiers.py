#!/usr/bin/env python
# coding: utf-8
import logging
import shutil
from pathlib import Path

import pytest

import pymasq.config as cfg
from pymasq.datasets import load_census
from pymasq.models.models import (
    LogisticRegressionClassifier,
    RFClassifier,
    TpotClassifier,
)
from pymasq.preprocessing import EmbeddingsEncoder, LabelEncoderPM

logger = logging.getLogger(__name__)


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
        (LogisticRegressionClassifier, LabelEncoderPM),
        (LogisticRegressionClassifier, EmbeddingsEncoder),
        (RFClassifier, LabelEncoderPM),
        (RFClassifier, EmbeddingsEncoder),
        (TpotClassifier, LabelEncoderPM),
        (TpotClassifier, EmbeddingsEncoder),
    ],
)
def test_classifiers(my_df, combo):
    classifier_type, preprocessor = combo
    logger.info(classifier_type)
    # check that the classifier gets the expected value given a set hmac key and set seed

    dir_name = "cache_test"
    Path(dir_name).mkdir(exist_ok=True)

    cfg.CACHE_HMAC_KEY = "my key"
    classifier = classifier_type(cache_location=dir_name, modeling_task="binary")

    kwargs = {}
    if classifier_type == TpotClassifier:
        kwargs = {"generations": 2, "population_size": 5, "cv": 2}

    # seed is an argument for embeddings only
    if preprocessor == EmbeddingsEncoder:
        classifier.train(
            df=my_df,
            y_column="sex",
            preprocessor=preprocessor,
            verbose=1,
            retrain=True,
            **kwargs,
        )
    else:
        classifier.train(
            df=my_df,
            y_column="sex",
            preprocessor=preprocessor,
            verbose=1,
            retrain=True,
            **kwargs,
        )
    # should make use of cache
    enc = preprocessor.encode(my_df, cache_location=dir_name, verbose=1)

    logger.info(type(enc.drop(["sex"], axis=1)))
    logger.info(type(enc.sex))
    score = classifier.predict(x_test=enc.drop(["sex"], axis=1), y_true=enc.sex)
    logger.info(f"{classifier.name}, {preprocessor}: {score}")
    assert 0.0 <= score <= 1.0, f"Score should be a valid AUC between 0 and 1: got {score}"

    # Check if the cached file loads, and that the hmac checks out
    logger.info(f"\n{classifier.name}, {preprocessor} load")
    classifier.load_trained_model(my_df, verbose=1)

    logger.info("removing cache")
    shutil.rmtree(dir_name)
