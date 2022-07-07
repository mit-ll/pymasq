from typing import Dict, Final, List, Optional, Tuple, Union

import logging

logger = logging.getLogger(__name__)

import numpy as np
from numpy import ndarray
import pandas as pd
import statsmodels.api as sm
import json
from boruta import BorutaPy
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype

from pymasq import BEARTYPE
from pymasq.preprocessing import EmbeddingsEncoder

__all__ = [
    "key_variable_exploration",
    "importance_scores",
    "random_forest_scores",
    "boruta_scores",
    "rfe_scores",
    "stepwise_scores",
    "stepwise_selection",
    "RANDOM_FOREST",
    "BORUTA",
    "RFE",
    "INCLUDE",
    "VARIABLE",
    "EMBEDDING",
    "RANKING",
    "EVIDENCE",
    "STEPWISE",
]


RANDOM_FOREST: Final = "Random_Forest"
BORUTA: Final = "Boruta"
RFE: Final = "RFE"
STEPWISE: Final = "Stepwise"
INCLUDE: Final = "Include"
VARIABLE: Final = "Variable"
EMBEDDING: Final = "Embedding"
RANKING: Final = "ranking"
EVIDENCE: Final = "evidence"


@BEARTYPE
def reverse_ranks(ranks: List[int]) -> List[int]:
    """
    Takes in a list of integer ranks and reverses the ranks.

    Parameters
    ----------
    ranks : List[int]
        List of integer ranks.

    Returns
    -------
    List[int]
        A list with the ranks reversed.
    """
    return [max(ranks) + 1 - r for r in ranks]


@BEARTYPE
def include_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a dataframe and returns the data sorted based on the "Include"
    column

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the binary label column and the other variables
        of interest.

    Returns
    -------
    pd.DataFrame
        A sorted dataframe.
    """
    include_dict = {"yes": 2, "maybe": 1, "no": 0}
    df["include_sort"] = df.Include.map(include_dict)
    df.sort_values(["include_sort", RANDOM_FOREST], ascending=[0, 0], inplace=True)
    df.drop("include_sort", axis=1, inplace=True)
    return df.reset_index(drop=True)


@BEARTYPE
def key_variable_exploration(
    df: pd.DataFrame,
    sensitive_col: str,
    preprocessed: bool = False,
    verbose: int = 0,
    num_components: int = 6,
    **kwargs: Dict[str, str],
) -> Dict[str, pd.DataFrame]:
    """
    Explore the ranking of key variables in a binary classification setting
    with the target column containing the binary labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the binary label column and the other variables of interest.

    sensitive_col : str
        Name of the column in the dataframe, `df`, that contains binary labels.

    preprocessed: bool, optional (Default: False, the data will be preprocessed)
        Flag for whether the dataframe passed in has already been preprocessed.

    verbose : int {0, 1, 2}, optional (Default: 0)
        Level of reporting from the algorithms:
            - 0 disables verbose logging.
            - 2 is step-by-step reporting.

    num_components : int, optional (Default: 6)
        Indicates the number of components used to represent categories
        variables, default is 6 so that each categories variable will be
        represented by six columns which will each contain a float element
        from a 6 dimensional vector.

        Note: The same category from a categories variable such as "blue" in
        a Color column would be represented by the same 6 numbers across the
        columns for every row in the original dataset that had the color "blue".

    **kwargs
        Additional arguments to be passed to `importance_Scores`:
        * methods : Tuple[str], optional Default: ('rf', 'boruta', 'rfe', 'stepwise')
            Names of the ranking methods to run.

    Returns
    -------
    Dict
        The dictionary contains a "ranking" dataframe with the aggregated
        ranking per variable and and "evidence" dataframe with the
        unaggreagted ranking for variable components.
    """

    progress_reporter = kwargs.get("callback", None)
    if progress_reporter is not None:
        progress_reporter(0.0)

    if preprocessed:
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns.values)
    else:
        numeric_cols = EmbeddingsEncoder.get_numerical_cols(df)
        df = EmbeddingsEncoder.encode(
            df,
            sensitive_col=sensitive_col,
            num_embedding_components=num_components,
            normalize=True,
        )

    methods = kwargs.get("methods", (RANDOM_FOREST, BORUTA, RFE, STEPWISE))
    categories = len(df[sensitive_col].dropna().unique())
    if categories < 2:
        print(
            "The kve function requires two categories for binary classification  and the {} column has {} class".format(
                sensitive_col, categories
            )
        )
        return dict()
    elif is_numeric_dtype(df[sensitive_col]):
        categories = -1

    rank_df = importance_scores(
        df, sensitive_col, categories=categories, verbose=verbose, **kwargs
    )
    if verbose > 0:
        print("Building ranking...")
    include_cols = [c for c in rank_df.columns if INCLUDE in c]

    rank_df[INCLUDE] = rank_df.apply(
        lambda r: "yes"
        if all(r[include_cols] == "yes")
        else "maybe"
        if any(r[include_cols] == "maybe") or any(r[include_cols] == "yes")
        else "no",
        axis=1,
    )
    rank_df = include_sort(rank_df)

    rank_df[EMBEDDING] = rank_df.Variable.apply(
        lambda x: "NA" if x in numeric_cols else x
    )
    rank_df.Variable = rank_df.Variable.apply(
        lambda x: x if x in numeric_cols else x.rsplit("_", 1)[0]
    )

    agg = {
        INCLUDE: lambda x: "yes"
        if all(x == "yes")
        else "maybe"
        if any(x == "maybe") or any(x == "yes")
        else "no"
    }
    if RANDOM_FOREST in methods:
        agg[RANDOM_FOREST] = "mean"
    ranking = rank_df.groupby([VARIABLE]).agg(agg)
    return {RANKING: ranking.round(4), EVIDENCE: rank_df.round(4)}


@BEARTYPE
def importance_scores(
    input_df: pd.DataFrame,
    sensitive_col: str,
    categories: int,
    methods: Optional[Tuple[str]] = None,
    verbose: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """
    Use Feature Selection methods to generate importance ranks for variables
    in a binary classification setting with the target variable being the
    binary labels.

    Parameters
    ----------
    input_df : pd.DataFrame
        Dataframe containing the binary label column and the other variables
        of interest.

    sensitive_col : str
        Name of the column in the dataframe that contains binary labels

    categories: int
        Number of categories in the senestive column used to determine the type
        of model used in feature selection, -1 indicates the column is continuous

    methods : Tuple[str], optional (Default: "Random_Forest","Boruta","RFE", "Stepwise")
        Names of the ranking methods to run

    verbose : int {0, 1, 2}, (Default: 0)
        Level of reporting from the algorithms:
            - 0 disables reporting
            - 2 is is step-by-step reporting

    Returns
    -------
    scores: pd.DataFrame
        The dataframe contains a column of all the variable names and columns
        of each of the ranking methods that are run on the data set
    """
    progress_reporter = kwargs.get(
        "callback", None
    )  # callable function that emits to main server
    if methods is None:
        methods = (RANDOM_FOREST, BORUTA, RFE, STEPWISE)
    method_len = float(len(methods))  # instantiated for progress emits
    method_count = 1  # instantiated for progress emits
    x_rf = input_df.drop([sensitive_col], axis=1)
    y_rf = input_df.loc[:, sensitive_col]
    x_train = x_rf.copy()
    y = y_rf
    score_dict = {}
    if RANDOM_FOREST in methods:
        if verbose > 0:
            print("Running Random Forest...")
        (
            score_dict[RANDOM_FOREST],
            score_dict[f"{RANDOM_FOREST}_{INCLUDE}"],
        ) = random_forest_scores(x_rf, y_rf, verbose=verbose, categories=categories)
        if progress_reporter is not None:
            progress_reporter(method_count / method_len)
            method_count += 1
    if BORUTA in methods and x_train.shape[0] >= 250:
        if verbose > 0:
            print("Running Boruta...")
        score_dict[f"{BORUTA}_{INCLUDE}"] = boruta_scores(
            x_train, y, verbose=verbose, categories=categories
        )
        if progress_reporter is not None:
            progress_reporter(method_count / method_len)
            method_count += 1
    if RFE in methods:
        if verbose > 0:
            print("Running Recursive Feature Elimination...")
        score_dict[f"{RFE}_{INCLUDE}"] = rfe_scores(
            x_train, y, verbose=verbose, categories=categories
        )
        if progress_reporter is not None:
            progress_reporter(method_count / method_len)
            method_count += 1
    if STEPWISE in methods:
        if verbose > 0:
            print("Running Stepwise...")
        score_dict[f"{STEPWISE}_{INCLUDE}"] = stepwise_scores(
            x_rf, y_rf, verbose=verbose
        )
        if progress_reporter is not None:
            progress_reporter(method_count / method_len)
            method_count += 1
    if not score_dict and x_train.shape[0] >= 2000:
        return pd.DataFrame()
    if progress_reporter is not None:
        progress_reporter(1.0)
    scores = pd.DataFrame(score_dict, index=x_rf.columns)
    scores.rename_axis(VARIABLE, inplace=True)
    scores.reset_index(inplace=True)
    return scores


@BEARTYPE
def random_forest_scores(
    x_train: pd.DataFrame,
    y: pd.Series,
    categories: int,
    n_estimators: int = 100,
    n_jobs: int = -1,
    random_state: int = 1234,
    verbose: int = 0,
) -> Tuple[ndarray, List[str]]:
    """
    Runs permutation importance on a Random Forest model which returns both
    the feature importances from the Random Forest model and well as a list
    of whether to include each column

    Parameters
    ----------
    x_train : pd.DataFrame
        A dataframe containing all input variables for training the model

    y : pd.Series
        A series containing the ground truth labels or numbers

    categories: int
        number of categories in the senestive column used to determine the type
        of model used in feature selection, -1 indicates the column is continuous

    n_estimators : int, optional (Default: 100)
        Number of trees that are constructed during the random forest

    n_jobs : int, optional(Default: -1)
        Number of workers to use for parallel processing
        -1 indicates use all available workers

    random_state: int, optional (Default: 1234)
        Integer seed for setting the random state in the model

    verbose : int {0, 1, 2}, optional (Default: 0)
        Level of reporting from the algorithms:
            - 0 disables verbose logging
            - 1 is is step-by-step reporting

    Returns
    -------
    Tuple
        Returns a 2-tuple containing a list of the feature importance scores
        from the OOB error when training the random forest model for each of
        the input features and a second list of strings, "yes" is a feature
        should be included as calculated by is the permutation mean is greater
        than 0 or "no" if not.
    """

    if categories >= 2:
        x_train, x_test, y_train, y_test = train_test_split(
            x_train, y, stratify=y, random_state=random_state
        )
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_train, y, random_state=random_state
        )
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )

    rf.fit(x_train, y_train)
    result = permutation_importance(
        rf, x_test, y_test, n_repeats=10, random_state=random_state, n_jobs=n_jobs
    )
    include = ["yes" if i > 0 else "no" for i in result.importances_mean]
    return rf.feature_importances_, include


@BEARTYPE
def boruta_scores(
    x_train: pd.DataFrame,
    y: pd.Series,
    categories: int,
    n_estimators: int = 1000,
    n_jobs: int = -1,
    random_state: int = 1234,
    verbose: int = 0,
    max_iter: int = 50,
) -> List[str]:
    """
    Boruta is an all relevant feature selection method, while most other are
    minimal optimal; this means it tries to find all features carrying
    information usable for prediction, rather than finding a possibly compact
    subset of features on which some classifier has a minimal error


    NOTE: Does not work with small data, requires >250 rows

    Parameters
    ----------
    x_train : pd.DataFrame
        A dataframe containing all input variables for training the model

    y : pd.Series
        A series containing the ground truth labels or numbers

    categories: int
        number of categories in the senestive column used to determine the type
        of model used in feature selection, -1 indicates the column is continuous

    n_estimators : int, optional (Default: 1000)
        Number of trees that are constructed during the random forest

    n_jobs : int, optional (Default: -1)
        Number of workers to use for parallel processing
            - -1 indicates use all available workers

    random_state: int, optional (Default: 1234)
        Integer seed for setting the random state in the model

    verbose : int {0, 1, 2}, optional (Default 2)
        Level of reporting from the algorithms:
            - 0 disables verbose logging
            - 2 is step-by-step reporting

    max_iter: int, optional (Default: 50)
        The number of maximum iterations to perform.

    Returns
    -------
    List[str]
        list of strings, contains whether a feature should be included in
        further analysis:
        - "yes": boruta ranking = 1
        - "maybe": boruta ranking = 2
        - "no": boruta ranking >= 3

    References
    ----------
    https://medium.com/@indreshbhattacharyya/feature-selection-categorical-feature-selection-boruta-light-gbm-chi-square-bf47e94e2558

    """
    if x_train.shape[0] < 250:
        print("Requires > 250 rows to be stable")
        return []
    if categories >= 2:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
    else:
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
    boruta_selector = BorutaPy(
        rf,
        verbose=verbose,
        n_estimators="auto",
        random_state=random_state,
        max_iter=max_iter,
    )
    if isinstance(x_train, np.ndarray):
        boruta_selector.fit(x_train, y)
    else:
        boruta_selector.fit(x_train.values, y.values)
    include = []
    for r in list(boruta_selector.ranking_):
        if r == 1:
            include.append("yes")
        elif r == 2:
            include.append("maybe")
        else:
            include.append("no")
    return include


@BEARTYPE
def rfe_scores(
    x_train: pd.DataFrame,
    y: pd.Series,
    categories: int,
    n_jobs: int = -1,
    random_state: int = 1234,
    verbose: int = 0,
    solver: str = "saga",
    penalty: str = "l2",
    l1_ratio: float = 0.5,
    step: int = 1,
    cv: int = 5,
) -> List[str]:
    """
    Feature ranking with recursive feature elimination.

    Given an external estimator that assigns weights to features
    (e.g., the coefficients of a linear model), the goal of recursive feature
    elimination (RFE) is to select features by recursively considering smaller
    and smaller sets of features. First, the estimator is trained on the
    initial set of features and the importance of each feature is obtained
    either through a coef_ attribute or through a feature_importances_
    attribute. Then, the least important features are pruned from current set
    of features. That procedure is recursively repeated on the pruned set
    until the desired number of features to select is eventually reached.

    Parameters
    ----------
    x_train : pd.DataFrame
        A dataframe containing all input variables for training the model

    y : pd.Series
        A series containing the ground truth labels or numbers

    categories: int
        number of categories in the senestive column used to determine the type
        of model used in feature selection, -1 indicates the column is continuous

    n_jobs : int, optional (Default: -1)
        Number of workers to use for parallel processing
        -1 indicates use all available workers

    random_state: int, optional (Default: 1234)
        Integer seed for setting the random state in the model


    verbose : int {0, 1, 2}, optional (Default: 0)
        Level of reporting from the algorithms:
            - 0 disables verbose logging
            - 2 is step-by-step reporting

    solver: str {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, optional (Default: "saga")
        Algorithm to use in the optimization problem.

        For small datasets, 'liblinear' is a good choice,
        whereas 'sag' and 'saga' are faster for large ones.

        For multiclass problems, only 'newton-cg', 'sag', 'saga' and
        'lbfgs' handle multinomial loss; 'liblinear' is limited to
        one-versus-rest schemes.

        'newton-cg', 'lbfgs', 'sag' and 'saga' handle L2 or no penalty

        'liblinear' and 'saga' also handle L1 penalty

        'saga' also supports 'elasticnet' penalty

        'liblinear' does not support setting penalty='none'

        Note that 'sag' and 'saga' fast convergence is only guaranteed on 0
        features with approximately the same scale. You can preprocess the
        data with a scaler from sklearn.preprocessing.

    penalty: {‘l1', ‘l2', ‘elasticnet', ‘none'}, optional (Default: "elasticnet")
        Used to specify the norm used in the penalization. The ‘newton-cg',
        ‘sag' and ‘lbfgs' solvers support only l2 penalties. ‘elasticnet' is
        only supported by the ‘saga' solver. If ‘none' (not supported by the
        liblinear solver), no regularization is applied.

    l1_ratio: float, optional (Default: 0.5)
        The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used
        if penalty='elasticnet'. Setting l1_ratio=0 is equivalent to using
        penalty='l2', while setting l1_ratio=1 is equivalent to using
        penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1
        and L2.

    step : int or float, optional (Default: 1)
        If greater than or equal to 1, then step corresponds to the (integer)
        number of features to remove at each iteration. If within (0.0, 1.0),
        then step corresponds to the percentage (rounded down) of features to
        remove at each iteration. Note that the last iteration may remove fewer
        than step features in order to reach min_features_to_select.

    cv : int, optional (Default: 5)
        Determines the cross-validation splitting strategy. Possible inputs
        for cv are:
            None, to use the default 5-fold cross-validation,
            integer, to specify the number of folds.

        For integer/None inputs, if y is binary or multiclass,
        sklearn.model_selection.StratifiedKFold is used.

        If the estimator is a classifier or if y is neither binary nor multiclass,
        sklearn.model_selection.KFold is used.

    Returns
    -------
    List[str]
        list of strings, contains whether a feature should be included in
        further analysis:
        - "yes": RFECV ranking = 1
        - "no": RFECV ranking not = 1
    """
    if categories == 2:
        if penalty == "elasticnet":
            estimator = LogisticRegression(
                random_state=random_state,
                solver=solver,
                penalty=penalty,
                l1_ratio=l1_ratio,
            )
        else:
            estimator = LogisticRegression(
                random_state=random_state, solver=solver, penalty=penalty
            )
    elif categories > 2:
        if penalty == "elasticnet":
            estimator = LogisticRegression(
                random_state=random_state,
                solver=solver,
                penalty=penalty,
                l1_ratio=l1_ratio,
                multi_class="ovr",
            )
        else:
            estimator = LogisticRegression(
                random_state=random_state,
                solver=solver,
                penalty=penalty,
                multi_class="ovr",
            )
    else:
        estimator = LinearRegression(normalize=True, n_jobs=n_jobs)
    rfecv_selector = RFECV(estimator, step=step, cv=cv, verbose=verbose, n_jobs=n_jobs)
    rfecv_selector.fit(x_train, y)
    return ["yes" if r == 1 else "no" for r in list(rfecv_selector.ranking_)]


@BEARTYPE
def stepwise_scores(
    x_train: pd.DataFrame,
    y: pd.Series,
    **kwargs: Dict[str, str],
) -> List[str]:
    """
    Feature ranking with stepwise selection with regression for retaining significant features.

    Built from the first answer from https://datascience.stackexchange.com/questions/24405/how-to-do-stepwise-regression-using-sklearn/24447#24447

    Parameters
    ----------
    x_train : pd.DataFrame
        A dataframe containing all input variables for training the model

    y : pd.Series
        A series containing the ground truth labels or numbers

    **kwargs
        Additional arguments to be passed to `stepwise_selection`.
        * initial_list : List[str], optional (Default: None)
            list of features to start with (column names of x_train)

        * threshold_in: float, optional (Default: 0.01)
            include a feature if its p-value < threshold_in

        * threshold_out: float, optional (Default: 0.05)
            exclude a feature if its p-value > threshold_out

        * iterations: int, optional (Default: 100)
            number of iterations for testing parameters to avoid
            infinite loops

        * verbose: bool, optional (Default: True)
            flag for whether to print the sequence of inclusions and exclusions


    Returns
    -------
    List[str]:
        list of strings, contains whether a feature should be included in
        further analysis:
            - "yes": stepwise_selection selects column given parameters
            - "no": stepwise_selection does not selects column given parameters
    """
    if not is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)
    selected_cols = stepwise_selection(x_train, pd.Series(y), **kwargs)
    return ["yes" if col in selected_cols else "no" for col in x_train.columns]


@BEARTYPE
def stepwise_selection(
    x_train: pd.DataFrame,
    y: pd.Series,
    initial_list: Optional[List[str]] = None,
    threshold_in: float = 0.01,
    threshold_out: float = 0.05,
    iterations: int = 100,
    verbose: Union[int, bool] = True,
):
    """
    Perform a forward-backward feature selection
    based on p-value from `statsmodels.api.OLS`.

    Parameters
    ----------
    x_train : pd.DataFrame
        A dataframe containing all input variables for training the model
        with candidate features

    y : pd.Series
        A series containing the ground truth labels or numbers

    initial_list : List[str], optional (Default: None)
        List of features to start with (column names of x_train)

    threshold_in: float, optional (Default: 0.01)
        Include a feature if its p-value < threshold_in

    threshold_out: float, optional (Default: 0.05)
        Exclude a feature if its p-value > threshold_out

    iterations: int, optional (Default: 100)
        Number of iterations for testing parameters to avoid
        infinite loops

    verbose: bool, optional (Default: True)
        Flag for whether to print the sequence of inclusions and exclusions

    Returns
    -------
    included: List[str]
        list of selected features

    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = []
    if initial_list:
        included = list(initial_list)
    count = 0
    tested = []

    # Resets number of iterations to width of the x_train dataframe so that each column can be tried
    if iterations < x_train.shape[1]:
        iterations = x_train.shape[1]
    while True and count < iterations:
        changed = False
        # forward step
        excluded = list(set(x_train.columns) - set(tested))
        excluded.sort()
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(
                y, sm.add_constant(pd.DataFrame(x_train[included + [new_column]]))
            ).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = excluded[new_pval.argmin()]
            included.append(best_feature)
            tested.append(best_feature)
            changed = True
            if verbose > 0:
                print("Add  {:30} with p-value {:.6}".format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x_train[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = included[pvalues.argmax()]
            included.remove(worst_feature)
            if verbose > 0:
                print("Drop {:30} with p-value {:.6}".format(worst_feature, worst_pval))
        if not changed:
            break
        count += 1
    return included
