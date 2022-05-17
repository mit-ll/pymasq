#!/usr/bin/env python
# coding: utf-8

import shutil
import pytest
import pymasq.config as cfg
from pathlib import Path
from pymasq.datasets import load_census
from pymasq.models.models import LogisticRegressionClassifier, RFClassifier
from pymasq.preprocessing import LabelEncoder_pm, EmbeddingsEncoder

# from pymasq.errors import InputError, DataTypeError

from pymasq.utils import cache


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
            LabelEncoder_pm,
            0.6,
            "cache_test/053cb5e57bfa9b5c9568625cb22588dd.larsCV.2bd270eec04828b035a1facfbb35f355.pkl",
            """larsCV. Description: Preprocessed with <class 'pymasq.preprocessing.preprocess.LabelEncoder_pm'>
First ten rows:
   age  fnlwgt  education  ...     sex capital_gain  income_level
0   39   77516  Bachelors  ...    Male         2174         <=50K
1   50   83311  Bachelors  ...    Male            0         <=50K
2   38  215646    HS-grad  ...    Male            0         <=50K
3   53  234721       11th  ...    Male            0         <=50K
4   28  338409  Bachelors  ...  Female            0         <=50K
5   37  284582    Masters  ...  Female            0         <=50K
6   49  160187        9th  ...  Female            0         <=50K
7   52  209642    HS-grad  ...    Male            0          >50K
8   31   45781    Masters  ...  Female        14084          >50K
9   42  159449  Bachelors  ...    Male         5178          >50K

[10 rows x 7 columns]""",
        ),
        (
            RFClassifier,
            EmbeddingsEncoder,
            0.5,
            "cache_test/053cb5e57bfa9b5c9568625cb22588dd.ENCV.e81a5b5eb0df48bc68540d7b71342a7d.pkl",
            """ENCV. Description: Preprocessed with <class 'pymasq.preprocessing.preprocess.EmbeddingsEncoder'>
First ten rows:
   age  fnlwgt  education  ...     sex capital_gain  income_level
0   39   77516  Bachelors  ...    Male         2174         <=50K
1   50   83311  Bachelors  ...    Male            0         <=50K
2   38  215646    HS-grad  ...    Male            0         <=50K
3   53  234721       11th  ...    Male            0         <=50K
4   28  338409  Bachelors  ...  Female            0         <=50K
5   37  284582    Masters  ...  Female            0         <=50K
6   49  160187        9th  ...  Female            0         <=50K
7   52  209642    HS-grad  ...    Male            0          >50K
8   31   45781    Masters  ...  Female        14084          >50K
9   42  159449  Bachelors  ...    Male         5178          >50K

[10 rows x 7 columns]""",
        ),
    ],
)
def test_cache(my_df, combo):
    classifier_type, preprocessor, answer, key, desc = combo
    print(classifier_type)

    dir_name = "cache_test"
    Path(dir_name).mkdir(exist_ok=True)

    cfg.CACHE_HMAC_KEY = "my key"
    classifier = classifier_type(cache_location=dir_name)
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
    enc = preprocessor.encode(my_df, cache_location=None)
    score = classifier.predict(x_test=enc.drop(["sex"], axis=1), y_true=enc.sex)
    print(f"{classifier.name}, {preprocessor}: {score}")
    assert round(score, 2) == answer, "Scores should match (trial {}) {} and {}".format(
        combo, score, answer
    )

    # Check if the cached file loads, and that the hmac checks out
    print(f"\n{classifier.name}, {preprocessor} load")
    classifier.load_trained_model(my_df, verbose=1)

    # Test that changing the hmac will cause a failure
    cfg.CACHE_HMAC_KEY = "not my key"
    try:
        classifier.load_trained_model(my_df)
        raise ("This test should have failed because the hmac key changed")
    except Exception as e:
        print("This error is a desired outcome of the test:")
        print("\t", e, "\n")
        pass

    cfg.CACHE_HMAC_KEY = "my key"
    # Assert to see if description was saved
    descriptions = cache.cache_info(dir_name)
    assert descriptions[key] == desc
    shutil.rmtree(dir_name)