import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from numpy import linspace
from typing import Optional

from pymasq.models import models
from pymasq.errors import InputError
from pymasq.metrics import utils
from pymasq.preprocessing import preprocess
from pymasq import BEARTYPE

__all__ = [
    "propensity_score",
    "proportion_diff_score",
    "jensen_shannon",
]


@BEARTYPE
def jensen_shannon(
    orig_df: pd.DataFrame,
    mod_df: pd.DataFrame,
    sensitive_col: Optional[str] = None,
    preprocessor: str = "embeddings",
) -> float:
    """
    Computes the Jensen-Shannon distance (not the divergence) of two data frames with same columns
    and dimensions. Results are between 1.0 to 0.0, where
        1.0 is 'there is complete difference between the data frames' and
        0.0 is 'there is no distinction between the two data frames.'

    Parameters
    ----------
    orig_df : pd.DataFrame
        A data frame that contains the original, unaltered data.

    mod_df : pd.DataFrame
        A data frame with the same shape and column names as orig_df, but with modified values

    sensitive_col : str
        A string for the column name in the data frame that contains binary labels

    preprocessor : str, optional (Default: "embeddings")
        A string indicating what pre-processor to use. Options are:
            "embeddings" (Default) uses pymasq.preprocessing
            "label_encode" uses pymasq.preprocessing.label_encode()
            None uses none (i.e., the data is already pre-processed)


    Returns
    -------
    float
        A distance between 0.0 and 1.0.

    """

    if preprocessor not in preprocess.preprocessor_fn:
        raise InputError("No such preprocessor defined in pymasq:", preprocessor)
    # Encode the two data frames (at once for consistent encodings)
    preprocessor_fn = preprocess.preprocessor_fn[preprocessor]
    orig_enc, mod_enc = preprocessor_fn.encode_both(
        df_A=orig_df, df_B=mod_df, sensitive_col=sensitive_col
    )

    # remove sensitive column
    if sensitive_col:
        orig_enc = orig_enc.drop([sensitive_col], axis=1)
        mod_enc = mod_enc.drop([sensitive_col], axis=1)
    # Compute JS on each column; store sum in js
    jensen_shannon = 0.0
    for col in orig_enc.columns:
        # we are going to create 100 bins between min_val and max_val
        min_val = min(orig_enc[col].min(), mod_enc[col].min())
        max_val = max(orig_enc[col].max(), mod_enc[col].max())
        # pd.cut() requires that min_val and max_val not be equal
        if max_val == min_val:
            max_val += 1
        # Makes a histogram/pmf for each data frame, based on 20 bins
        # Assumes all cells are numeric
        a1 = (
            pd.cut(
                orig_enc[col],
                bins=[x for x in linspace(min_val, max_val, 100)],
                include_lowest=True,
            )
            .value_counts(normalize=True, sort=False)
            .rename_axis("val")
            .reset_index(name="cnts")
        )
        a2 = (
            pd.cut(
                mod_enc[col],
                bins=[x for x in linspace(min_val, max_val, 100)],
                include_lowest=True,
            )
            .value_counts(normalize=True, sort=False)
            .rename_axis("val")
            .reset_index(name="cnts")
        )
        combined = a1.merge(a2, on="val", how="outer")
        jensen_shannon += distance.jensenshannon(
            combined[["cnts_x"]].fillna(0),
            combined[["cnts_y"]].fillna(0),
            base=2.0,  # must be base 2.0 or results are not between 0 and 1.
        )[0]
    # return the average JS score over all columns as if they are independent.
    return jensen_shannon / len(orig_enc.columns)


@BEARTYPE
def propensity_score(
    orig_df: pd.DataFrame,
    mod_df: pd.DataFrame,
    sensitive_col: str,
    test_size: float = 0.1,
    random_state: int = 1234,
    preprocessor: Optional[str] = "embeddings",
    method: str = "encv",
) -> float:
    """
    Uses elasticnet to train an optimized logisitic regression model to classify the difference between the original and
     modified data frames and returns the Area Under the ROC Curve (AUC) normalized to between 1.0 to 0.0, where 1.0 is 'there
     is complete difference between the data frames' and 0.0 is 'there is no distinction between the two data frames.'

    This function trains a classifier where class 0 is the orig_df and class 1 is the mod_df. The classifier is trained
    based on a  fraction of the two dataframes and then tested on the remaining fraction. The AUC score for the prediction
    is returned as the score.

    How is this different than AUC_Score? AUC Score trains a classifier based on orig_df in order to predict the values in sensitive_col from
    the values in the other columns. It then applies the classifier to the mod_df --- that is, based on the
    values in the other columns in mod_df, it uses the trained classifier to predict the values in the sensitve_column
    of mod_df. The AUC score for the prediction is returned as the score.

    Propensity stacks the orig_df and mod_df in one combined df. It then adds a
    0/1 column that specifies the class of the row original. The combined frame is split into train and test. The classifer
    is trained to determine the class of each row. The classifier is then tested. Propensity is the AUC score for the prediction.)

    AUC_SCORE:
        Train(orig, predict[sensitive]) ->
            - Test(orig, predict[sensity])
            - Test(mod, predict[sensity])
        Difference in test performance is AUC_Score

    PROPENSITY:
        combined = orig+mod with new column
        traindf = x% of combined
        testdf = (1-x)% of combined
        Classifer(traindf, predict[new column])
            - Test(testdf, predict[new column])
        Propensity of the area under the curve from Test()ing

    1.0 indicates high disimilarity between orig_df and mod_df
    0.0 indicates high similarity between orig_df and mod_df

    Parameters
    ----------
    orig_df : pd.DataFrame
        A data frame that contains the original, unaltered data.

    mod_df : pd.DataFrame
        A data frame with the same shape and column names as orig_df, but the values. Must have the same columns as orig_df.

    sensitive_col : str
        A string for the column name in the data frame that contains binary labels

    n_jobs : int, optional (Default: -1)
        Number of workers to use for parallel processing
            -1 indicates use all available workers

    random_state: int, optional (Default: 1234)
        Integer seed for setting the random state in the model

    preprocessor : str, optional (Default: "embeddings")
        A string indicating what pre-processor to use. Options are:
            "embeddings" (Default) uses pymasq.preprocessing.preprocess_data()
            "label_encode" uses pymasq.preprocessing.label_encode()
            None uses none (i.e., the data is already pre-processed)

    method : str, optional (Default: "encv")
        A string indicating what classier to use. Options are:
            "envc" (Default) uses sklearn's ElasticNetVC()
            "larscv" uses sklearn's LarsCV()

    Returns
    -------
    float
        The normalized reverse AUC score between 0.0 and 1.0

    """

    # If we allowed TPOT, it would retrain from scratch at every iteration (too slow)
    if method == "tpot":
        raise InputError("TPOT is not an option for propensity_score().")

    if method not in models.model_fn:
        raise InputError("No such classifier defined in pymasq:", method)
    if preprocessor not in preprocess.preprocessor_fn:
        raise InputError("No such preprocessor defined in pymasq:", preprocessor)

    # Encode the two data frames (at once for consistent encodings)
    preprocessor_fn = preprocess.preprocessor_fn[preprocessor]
    orig_enc, mod_enc = preprocessor_fn.encode_both(
        df_A=orig_df, df_B=mod_df, sensitive_col=sensitive_col
    )
    # Create a unique column name to mark from which dataframe a row came from
    class_col = utils.uniq_col_name(orig_df)
    orig_enc[class_col] = 0
    mod_enc[class_col] = 1
    # Make one dataframe that has all rows
    comb_df = pd.concat(
        [orig_enc, mod_enc],
        axis=0,
        sort=False,
    ).reset_index(drop=True)

    # set up variables needed for test and train etc
    y = comb_df[class_col]
    if sensitive_col:
        x_train = comb_df.drop([class_col, sensitive_col], axis=1)
    else:
        x_train = comb_df.drop([class_col], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        x_train, y, test_size=test_size, random_state=random_state
    )
    # allocate a classifer, train, and predict
    classifer_fn = models.model_fn[method]()
    classifer_fn.train(df=X_train.join(y_train), y_column=class_col, preprocessor=None)
    auc = classifer_fn.predict(X_test, y_true=y_test)

    # 0.5 is a good score here and so we normalize to 1.0
    # e.g., if we pass in identical DF's, the AUC score should be 0.5, hence we return 0.0 (because 0.0 best score)
    # if we pass in two completely different DFs, the AUC should be 1.0, hence we return 1.0 (because 1.0 is worst score)
    # we call abs() because complement of classifier works for us. i.e., AUC of 0.0 returns 1.0 as worst scoore (because 0.0 is worst)
    return abs(auc - 0.5) / 0.5


@BEARTYPE
def proportion_diff_score(orig_df: pd.DataFrame, mod_df: pd.DataFrame):
    """
    Calculates the proportion of cells that are different (i.e., not equal) between two data
    frames and returns a proportion between 1.0 to 0.0.

    1.0 indicates high disimilarity between orig_df and mod_df
    0.0 indicates high similarity between orig_df and mod_df

    Parameters
    ----------
    orig_df : pd.DataFrame
        A data frame that contains the original, unaltered data.

    mod_df : pd.DataFrame
        A data frame with the same shape and column names as orig_df, but the values

    Returns
    -------
    float
        The proportion difference score between 0.0 and 1.0.

    """
    if orig_df.shape != mod_df.shape:
        raise InputError(
            "The two data frames are not the same dimensions, orignal data frame is {} and the modified data frame is {}".format(
                orig_df.shape, mod_df.shape
            )
        )
    n = orig_df.shape[0]
    m = orig_df.shape[1]
    return sum((orig_df != mod_df).sum(axis=0)) / (n * m)
