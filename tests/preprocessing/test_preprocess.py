#!/usr/bin/env python
# coding: utf-8

import pytest

from numpy import NaN

from pymasq.datasets import load_census

from pymasq.preprocessing import embed_entities, LabelEncoder_pm, EmbeddingsEncoder

# from pymasq.errors import InputError, DataTypeError


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
    df = df.loc[:10, cols]
    return df


# ----- Method: Index Tests -----
def test_embed_entites_1(my_df):
    """
    Tests that embed_entities returns the correct arrays for each marital_status category given a binary target column in a series.
    """
    ret = embed_entities(
        my_df["sex"], my_df[["marital_status"]], cache_location=None, retrain=True
    )
    assert my_df["marital_status"].nunique() == ret["marital_status"].shape[0]


def test_embed_entites_2(my_df):
    """
    Tests that embed_entities returns arrays for each education category given a binary target column.
    """
    ret = embed_entities(
        my_df["sex"], my_df[["education"]], cache_location=None, retrain=True
    )
    assert my_df["education"].nunique() == ret["education"].shape[0]


def test_embed_entites_3(my_df):
    """
    Tests that embed_entities returns arrays for each education category given a categorical target column.
    """
    ret = embed_entities(
        my_df["marital_status"], my_df[["education"]], cache_location=None, retrain=True
    )
    assert my_df["education"].nunique() == ret["education"].shape[0]


def test_embed_entites_4(my_df):
    """
    Tests that embed_entities returns arrays for each education category given a categorical target column,
    with two columns.
    """
    ret = embed_entities(
        my_df["marital_status"],
        my_df[["education"]],
        num_embedding_components=2,
        cache_location=None,
        retrain=True,
    )
    assert my_df["education"].nunique() == ret["education"].shape[0]


def test_embed_entites_5(my_df):
    """
    Tests that embed_entities returns arrays for each education category given a categorical target column,
    with 10 columns.
    """
    ret = embed_entities(
        my_df["marital_status"],
        my_df[["education"]],
        num_embedding_components=10,
        cache_location=None,
        retrain=True,
    )
    assert my_df["education"].nunique() == ret["education"].shape[0]


def test_embed_entites_6(my_df):
    """
    Tests that embed_entities returns the correct arrays for each marital_status category given a binary target column in a data frame.
    """
    ret = embed_entities(
        my_df[["sex"]], my_df[["marital_status"]], cache_location=None, retrain=True
    )
    assert my_df["marital_status"].nunique() == ret["marital_status"].shape[0]


def test_embed_entites_7(my_df):
    """
    Tests that embed_entities returns the correct arrays for each marital_status category given a numeric target column in a data frame.
    """
    ret = embed_entities(
        my_df["age"], my_df[["marital_status"]], cache_location=None, retrain=True
    )
    assert my_df["marital_status"].nunique() == ret["marital_status"].shape[0]


# def test_embed_entites_8(my_df):
#     """
#     Tests that embed_entities returns arrays for each education category given two target columns.
#     """
#     ret = embed_entities(my_df[["sex", "marital_status"]], my_df[["education"]])
#     print(my_df["education"].nunique())
#     print(ret["education"].shape[0])
#     assert my_df["education"].nunique() == ret["education"].shape[0]


def test_embed_entites_9(my_df):
    """
    Tests that embed_entities returns arrays for each education category given a numeric target column
    """
    ret = embed_entities(
        my_df["fnlwgt"],
        my_df[["education"]],
        cache_location=None,
        retrain=True,
    )
    print(my_df["education"].nunique())
    print(ret["education"].shape[0])
    assert my_df["education"].nunique() == ret["education"].shape[0]


def test_preprocess_data_1(my_df):
    """
    Tests that preprocess_data six columns to replace the single marital_status categorical column.
    """
    ret = EmbeddingsEncoder.encode(
        my_df[["sex", "marital_status"]],
        sensitive_col="sex",
        retrain=True,
        cache_location=None,
    )
    assert ret.shape[1] == 7


def test_preprocess_data_2(my_df):
    """
    Tests that preprocess_data returns 7 unique values for 7 marital_status categories.
    """
    ret = EmbeddingsEncoder.encode(
        my_df[["sex", "marital_status"]],
        sensitive_col="sex",
        retrain=True,
        cache_location=None,
    )
    assert (
        my_df["marital_status"].nunique()
        == ret.groupby("marital_status_0").count().shape[0]
    )


def test_preprocess_data_3(my_df):
    """
    Tests that preprocess_data returns 7 unique values for 7 marital_status categories.
    """
    ret = EmbeddingsEncoder.encode(
        my_df[["age", "marital_status"]],
        sensitive_col="age",
        retrain=True,
        cache_location=None,
    )
    assert (
        my_df["marital_status"].nunique()
        == ret.groupby("marital_status_0").count().shape[0]
    )


def test_preprocess_data_4(my_df):
    """
    Tests that preprocess_data includes income_level on numeric_cols list
    """
    ret = EmbeddingsEncoder.encode(
        my_df,
        sensitive_col="age",
        retrain=True,
        cache_location=None,
    )
    assert "income_level" in ret


def test_missing_data_1(my_df):
    """
    Tests that missing data in the sensitive column will throw a Value Error in preprocessing
    """
    my_df.at[2, "sex"] = NaN
    with pytest.raises(ValueError):
        ret = EmbeddingsEncoder.encode(
            my_df,
            sensitive_col="sex",
            retrain=True,
            cache_location=None,
        )


###################################################################


def test_label_encode_1(my_df):
    """
    test that labels are not reshuffled in columns and that columns are not re-arranged in the data frame

    my_df
        age  fnlwgt     education         marital_status     sex  capital_gain income_level
    0    39   77516     Bachelors          Never-married    Male          2174        <=50K
    1    50   83311     Bachelors     Married-civ-spouse    Male             0        <=50K
    2    38  215646       HS-grad               Divorced    Male             0        <=50K
    3    53  234721          11th     Married-civ-spouse    Male             0        <=50K
    4    28  338409     Bachelors     Married-civ-spouse  Female             0        <=50K
    5    37  284582       Masters     Married-civ-spouse  Female             0        <=50K
    6    49  160187           9th  Married-spouse-absent  Female             0        <=50K
    7    52  209642       HS-grad     Married-civ-spouse    Male             0         >50K
    8    31   45781       Masters          Never-married  Female         14084         >50K
    9    42  159449     Bachelors     Married-civ-spouse    Male          5178         >50K
    10   37  280464  Some-college     Married-civ-spouse    Male             0         >50K

    enc
        age  fnlwgt  education  marital_status  sex  capital_gain  income_level
    0    39   77516          2               3    1          2174             0
    1    50   83311          2               1    1             0             0
    2    38  215646          3               0    1             0             0
    3    53  234721          0               1    1             0             0
    4    28  338409          2               1    0             0             0
    5    37  284582          4               1    0             0             0
    6    49  160187          1               2    0             0             0
    7    52  209642          3               1    1             0             1
    8    31   45781          4               3    0         14084             1
    9    42  159449          2               1    1          5178             1
    10   37  280464          5               1    1             0             1

    """
    enc, _ = LabelEncoder_pm.encode_both(my_df, my_df)
    assert (
        enc.to_json()
        == '{"age":{"0":39,"1":50,"2":38,"3":53,"4":28,"5":37,"6":49,"7":52,"8":31,"9":42,"10":37},"fnlwgt":{"0":77516,"1":83311,"2":215646,"3":234721,"4":338409,"5":284582,"6":160187,"7":209642,"8":45781,"9":159449,"10":280464},"education":{"0":2,"1":2,"2":3,"3":0,"4":2,"5":4,"6":1,"7":3,"8":4,"9":2,"10":5},"marital_status":{"0":3,"1":1,"2":0,"3":1,"4":1,"5":1,"6":2,"7":1,"8":3,"9":1,"10":1},"sex":{"0":1,"1":1,"2":1,"3":1,"4":0,"5":0,"6":0,"7":1,"8":0,"9":1,"10":1},"capital_gain":{"0":2174,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":14084,"9":5178,"10":0},"income_level":{"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":1,"8":1,"9":1,"10":1}}'
    )


def test_label_encode_2(my_df):
    """
    test that if a value appears in one df but not the other, it's given a unique label across both dataframes.

    my_df1
    age  fnlwgt         education         marital_status     sex  capital_gain  income_level
    0   39   77516      Bachelors          Never-married    Male          2174         <=50K
    2   38  215646        HS-grad               Divorced    Male             0         <=50K
    8   31   45781        Masters          Never-married  Female         14084          >50K

    my_df2
        age  fnlwgt     education         marital_status     sex  capital_gain  income_level
    1    50   83311     Bachelors     Married-civ-spouse    Male             0         <=50K
    3    53  234721          11th     Married-civ-spouse    Male             0         <=50K
    4    28  338409     Bachelors     Married-civ-spouse  Female             0         <=50K
    5    37  284582       Masters     Married-civ-spouse  Female             0         <=50K
    6    49  160187           9th  Married-spouse-absent  Female             0         <=50K
    7    52  209642       HS-grad     Married-civ-spouse    Male             0          >50K
    9    42  159449     Bachelors     Married-civ-spouse    Male          5178          >50K
    10   37  280464  Some-college     Married-civ-spouse    Male             0          >50K

    enc1
        age  fnlwgt   education           marital_status     sex  capital_gain  income_level
    0    39   77516          2                         3       1          2174             0
    2    38  215646          3                         0       1             0             0
    8    31   45781          4                         3       0         14084             1

    enc2
        age  fnlwgt  education            marital_status     sex  capital_gain  income_level
    1    50   83311          2                         1       1             0             0
    3    53  234721          0                         1       1             0             0
    4    28  338409          2                         1       0             0             0
    5    37  284582          4                         1       0             0             0
    6    49  160187          1                         2       0             0             0
    7    52  209642          3                         1       1             0             1
    9    42  159449          2                         1       1          5178             1
    10   37  280464          5                         1       1             0             1

    """
    my_df1 = my_df[my_df["marital_status"].isin(["Never-married", "Divorced"])]
    my_df2 = my_df[~my_df["marital_status"].isin(["Never-married", "Divorced"])]
    enc1, enc2 = LabelEncoder_pm.encode_both(my_df1, my_df2)
    assert set(enc1.marital_status).isdisjoint(set(enc2.marital_status))
    assert (
        enc1.to_json()
        == '{"age":{"0":39,"2":38,"8":31},"fnlwgt":{"0":77516,"2":215646,"8":45781},"education":{"0":2,"2":3,"8":4},"marital_status":{"0":3,"2":0,"8":3},"sex":{"0":1,"2":1,"8":0},"capital_gain":{"0":2174,"2":0,"8":14084},"income_level":{"0":0,"2":0,"8":1}}'
    )
    assert (
        enc2.to_json()
        == '{"age":{"1":50,"3":53,"4":28,"5":37,"6":49,"7":52,"9":42,"10":37},"fnlwgt":{"1":83311,"3":234721,"4":338409,"5":284582,"6":160187,"7":209642,"9":159449,"10":280464},"education":{"1":2,"3":0,"4":2,"5":4,"6":1,"7":3,"9":2,"10":5},"marital_status":{"1":1,"3":1,"4":1,"5":1,"6":2,"7":1,"9":1,"10":1},"sex":{"1":1,"3":1,"4":0,"5":0,"6":0,"7":1,"9":1,"10":1},"capital_gain":{"1":0,"3":0,"4":0,"5":0,"6":0,"7":0,"9":5178,"10":0},"income_level":{"1":0,"3":0,"4":0,"5":0,"6":0,"7":1,"9":1,"10":1}}'
    )
