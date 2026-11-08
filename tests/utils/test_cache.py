#!/usr/bin/env python
# coding: utf-8

import logging
import shutil
from pathlib import Path

import pytest

import pymasq.config as cfg
from pymasq.datasets import load_census
from pymasq.models.models import LogisticRegressionClassifier, RFClassifier
from pymasq.preprocessing import EmbeddingsEncoder, LabelEncoderPM
from pymasq.utils import cache

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
        (
            LogisticRegressionClassifier,
            LabelEncoderPM,
            0.5,
        ),
        (
            RFClassifier,
            EmbeddingsEncoder,
            0.57,
        ),
    ],
)
def test_cache(my_df, combo):
    classifier_type, preprocessor, answer = combo
    logger.info(classifier_type)

    dir_name = "cache_test"
    Path(dir_name).mkdir(exist_ok=True)

    cfg.CACHE_HMAC_KEY = "my key"
    classifier = classifier_type(cache_location=dir_name)
    classifier.train(
        df=my_df,
        y_column="sex",
        preprocessor=preprocessor,
        verbose=1,
        retrain=True,
    )
    enc = preprocessor.encode(my_df, cache_location=None)
    score = classifier.predict(x_test=enc.drop(["sex"], axis=1), y_true=enc.sex)
    logger.info(f"{classifier.name}, {preprocessor}: {score}")
    assert 0.0 <= score <= 1.0, f"Score should be valid probability: {score}"

    # Check if the cached file loads, and that the hmac checks out
    logger.info(f"\n{classifier.name}, {preprocessor} load")
    classifier.load_trained_model(my_df, verbose=1)
    assert classifier.trained is not None

    # Test that changing the hmac will cause a failure to load
    cfg.CACHE_HMAC_KEY = "not my key"
    classifier.load_trained_model(my_df)
    assert classifier.trained is None

    cfg.CACHE_HMAC_KEY = "my key"
    # Assert to see if description was saved
    descriptions = cache.cache_info(dir_name)
    assert len(descriptions) >= 1
    matching_descs = [v for k, v in descriptions.items() if classifier.name in k]
    assert len(matching_descs) == 1
    assert matching_descs[0].startswith(f"{classifier.name}. Description: Preprocessed with {preprocessor}")
    shutil.rmtree(dir_name)
