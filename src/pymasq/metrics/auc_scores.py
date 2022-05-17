from pymasq.metrics.utils import _get_model_task
from pymasq.config import CLASSIFIER_MODELS, REGRESSOR_MODELS
import numpy as np
import pandas as pd
from typing import Optional, Union
from sklearn.preprocessing import LabelEncoder as skLabelEncoder

from pymasq.models import models
from pymasq.preprocessing import preprocess
from pymasq.errors import InputError
from pymasq import BEARTYPE

__all__ = ["auc_score"]


@BEARTYPE
def auc_score(
    orig_df: pd.DataFrame,
    mod_df: pd.DataFrame,
    sensitive_col: str,
    method: str,
    preprocessor: str = "embeddings",
    modeling_task: Optional[str] = None,
    cache_location: Optional[str] = None,
    absolute_risk: bool = False,
    retrain: bool = True,
    **kwargs
) -> float:
    """
    Calculates the AUC score for the provided original and modified dataframes given the target
    column name and cache folder.

    AUC Score trains a classifier based on orig_df in order to predict the values in sensitive_col from
    the values in the other columns. It then applies the classifier to the mod_df --- that is, based on the
    values in the other columns in mod_df, it uses the trained classifier to predict the values in the sensitve_column
    of mod_df. The AUC score for the prediction is returned as the score.

    How is this different than propensity? Propensity stacks the orig_df and mod_df in one combined df. It then adds a
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
    orig_df : pd.Dataframe
        Data frame containing the original unaltered data
    mod_df : pd.Dataframe
        Data frame containing the modified data after application of mitigation(s)
    sensitive_col : str
        String of sensitive column name in the data frame that contains labels
    method : str
        The classifier method to use. There are several options.
        - "encv" denotes the sklearn ElasticNetCV regressor
        - "larscv" denotes the sklearn LarsCV regressor
        - "rfreg" denotes the sklearn Random Forest regressor
        - "logreg" denotes the sklearn Logisitc RegressionCV classifier
        - "rfclass" denotes the sklearn Random Forest classifier
        - "tpotreg" denotes the use of a pre-trained TPOT regressor. To pre-train
          TPOT call pymasq.models.TpotRegressor.train(), which will save the trained
          model to a cache location. If a different than the default pymasq location is
          used, specify the path in the "cache_location" argument. If method is "tpot",
          then "tpot_fname" should be passed as a named keyword argument.
        - "tpotclass" denotes the use of a pre-trained TPOT classifer. To pre-train
          TPOT call pymasq.models.TpotClassifier.train(), which will save the trained
          model to a cache location. If a different than the default pymasq location is
          used, specify the path in the "cache_location" argument. If method is "tpot",
          then "tpot_fname" should be passed as a named keyword argument.
    preprocessor : str, optional (Default: 'embeddings')
        The pymasq preprocessor to use. There are several options.
        - 'embeddings': advanced processing of numeric and categorical data.
        - 'label_encode': skLearn-based encoding of categorical data only
        - None: the dataframes are not processed and assumed ready for use by a classifier
    modeling_task : str, optional (Default: None)
        Type of modeling used to prepare model and preprocessor, if not set then it
        will calculated from the sensitive column
    cache_location :  str, optional (Default: None)
        String for the directory path to the cache for the current TPOT data
    absolute_risk : bool, optional (Default: False)
        Boolean flag that, when true, calculates the absolute risk score.
    retrain: boolean, optional (Default: True)
        Whether to ignore cached training data
    **kwargs: Dict[Any, Any]
        Values for algorithm-specific headers to be added to called method.
    Returns
    -------
    float
        The auc score, as absolute risk or as a risk score

    """
    if method not in models.model_fn:
        raise InputError("No such classifier defined in pymasq:", method)
    if preprocessor not in preprocess.preprocessor_fn:
        raise InputError("No such preprocessor defined in pymasq:", preprocessor)

    # Tests if the sensitive variable is categorical or numeric and stores value in
    # classifier_bool
    if modeling_task is None:
        modeling_task = _get_model_task(orig_df[sensitive_col])

    if modeling_task in ("binary", "multi_class"):
        classifer_fn = models.model_fn[method](cache_location)
        preprocessor_fn = preprocess.preprocessor_fn[preprocessor]
    ###TODO Add Regression into AUC_Score
    elif modeling_task == "regression":
        raise InputError(
            "Sensitive Column: {} dtype ({}) does not match modeling task: {} with method: {}".format(
                sensitive_col, orig_df[sensitive_col].dtypes, "regression", method
            )
        )
    else:
        raise InputError(
            "Sensitive Column: {} dtype ({}) does not match modeling task: {} with method: {}".format(
                sensitive_col,
                orig_df[sensitive_col].dtypes,
                modeling_task,
                method,
            )
        )
    # Encode the two data frames (at once for consistent encodings)
    orig_enc, mod_enc = preprocessor_fn.encode_both(
        df_A=orig_df, df_B=mod_df, sensitive_col=sensitive_col
    )
    # Train the classifer based on only the original data
    classifer_fn.train(
        df=orig_enc,
        preprocessor=preprocessor_fn,
        y_column=sensitive_col,
        retrain=retrain,
        **kwargs
    )
    assert classifer_fn.trained is not None

    # Check how well our classifier, trained on original df (and now using
    # either the original df or the modified df) can predict values
    # in the sensitive column from the values in the other columns.
    scores = {}
    for df, dtype in [(orig_enc, "orig"), (mod_enc, "mod")]:
        x_test = df.drop(sensitive_col, axis=1)
        y = pd.Series(skLabelEncoder().fit_transform(df[sensitive_col]))
        scores[dtype] = classifer_fn.predict(x_test=x_test, y_true=y)

    # Report how well it did in one of these two formats
    orig_score = abs(0.5 - scores["orig"])
    mod_score = abs(0.5 - scores["mod"])
    if absolute_risk:
        risk_score = mod_score / 0.5
    else:
        risk_score = min(orig_score, mod_score) / orig_score
    return risk_score
