from pymasq.config import DEFAULT_LOGISITIC_REGRESSION_SOLVER, DEFAULT_SEED
import pandas as pd
import numpy as np
from typing import List, Optional, Type, Any, Union
from sklearn.preprocessing import LabelEncoder
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import ElasticNetCV, ElasticNet, LarsCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pymasq import BEARTYPE
from pymasq.models._base import ModelingBase
from pymasq.preprocessing._base import PreprocessorBase

# In this file:
# Five children of ModelingBase
#
# 1. LarsCvRegressor
# 2. ElasticNetCvRegressor
# 3. TpotRegressor
# 4. Classifier
# 5. TpotClassifier


#########################


def mape(
    y_true: Union[pd.Series, List[float]], y_pred: Union[pd.Series, List[float]]
) -> float:
    """MAPE can be considered as a loss function to define the error termed by the model evaluation. Using MAPE,
    we can estimate the accuracy in terms of the differences in the actual v/s estimated values.

    Parameters
    ----------
    y_true : array-like
        List of ground truth values from continuous variable
    predicted : array-like
        List of predicted values from continuous variable

    Returns
    -------
    mape: float
        Percent error of the predicted to the actual 0.0 and 1.0

    """
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


class LarsCvRegressor(ModelingBase):
    @BEARTYPE
    def __init__(self, cache_location: Optional[str] = None):
        """
        Create an instance of sklearn's LarsCV regressor.

        Parameters
        ----------
        cache_location : str, optional (Default: None)
            A string for the directory path to the cache for the current data.

        """
        super().__init__(name="larscv", cache_location=cache_location)

    @BEARTYPE
    def train(
        self,
        df: pd.DataFrame,
        y_column: str,
        preprocessor: Optional[Type[PreprocessorBase]],
        retrain: bool = False,
        verbose: int = 0,
        **kwargs,
    ):
        """
        Train sklearn's LarsCV regressor. The trained model is stored in LarsCvRegressor.cache_location.

        Parameters
        ----------
        df : pd.Dataframe
            A data frame containing the data to train the model including a numeric column for regression
        y_column : str
            The column name in the data frame that contains continuous values of ground truth
        preprocessor : PreprocessorBase
            A child of PreprocessorBase class indicating what preprocessor to use. Options are:
            - pymasq.preprocessing.EmbeddingsEncoder
            - pymasq.preprocessing.LabelEncoder_pm
            - None (i.e., the data is already pre-processed)
        retrain: bool (Default: False)
            Ignore cached results and retrain
        verbose: in (Default 0)
            0 for no verbose output
            1 for verbose logging


        """
        # this call to super() loads the cache if it exists, and then returns a  pre-processed version for training.
        # some preprocessors are not expecting a seed kw argument.
        df_enc = super().train(
            df, y_column, preprocessor, retrain, verbose, seed=self.seed
        )

        # we found it in the cache
        if self.trained is not None:
            return

        # We didn't: train the model and cache it.
        if verbose > 0:
            print("Training LarsCV model ")
        x_train = df_enc.drop(y_column, axis=1)
        y = df_enc[y_column]
        self.trained = LarsCV(n_jobs=self.n_jobs)
        self.trained.fit(x_train, y)

        # save to cache
        self.save_trained_model(
            df, verbose=verbose, description="Preprocessed with " + str(preprocessor)
        )

    @BEARTYPE
    def predict(self, x_test: pd.DataFrame, y_true: pd.Series) -> float:
        """
        Using an already trained model, predicts values in y_true. The Mean Absolute Pecent Error result is returned.

        Note: this function assumes that df is pre-processed as is typical for pymasq internal usage.

        Parameters
        ----------
        x_test : pd.Dataframe
            A data frame containing the data to train the model including a label column for binary classification
        y_true : pd.Series
            A pandas series contains continuous values as ground truth

        Returns
        -------
        float:
            The MAPE score, a value between 0.0 and 1.0

        """
        assert self.trained is not None

        y_predict = self.trained.predict(x_test)
        return mape(y_true=y_true, y_pred=y_predict)


#########################
class ElasticNetCvRegressor(ModelingBase):
    @BEARTYPE
    def __init__(self, cache_location: Optional[str] = None):
        """
        Create an instance of sklearn's ElasticNetCV classifier.

        Parameters
        ----------
        cache_location : str, optional (Default: None)
            A string for the directory path to the cache for the current data.

        """
        super().__init__(name="encv", cache_location=cache_location)

    @BEARTYPE
    def train(
        self,
        df: pd.DataFrame,
        y_column: str,
        preprocessor: Optional[Type[PreprocessorBase]],
        retrain: bool = False,
        verbose: int = 0,
        **kwargs,
    ):
        """
        Train based on the given dataframe. Model is saved to the cache location.

        Parameters
        ----------
        df : pd.Dataframe
            A data frame containing the data to train the model including a numeric column for regression
        y_column : str
            The column name in the data frame that contains numeric values of ground truth
        preprocessor : PreprocessorBase  (Default: None)
            A child of PreprocessorBase class indicating what preprocessor to use. Options are:
            - pymasq.preprocessing.EmbeddingsEncoder
            - pymasq.preprocessing.LabelEncoder_pm
            - None (i.e., the data is already pre-processed)
        retrain: bool (Default: False)
            Ignore cached results and retrain
        verbose: int (Default 0)
            0 for no verbose output
            1 for verbose output
        """
        # super().train() loads cache if it exists, and returns pre-processed df for training
        df_enc = super().train(
            df, y_column, preprocessor, retrain, verbose, seed=self.seed
        )
        # we found a cache file
        if self.trained is not None:
            return

        # no cache found, we need to train.
        if verbose > 0:
            print("Training ElasticNetCV model ")
        x_train = df_enc.drop(y_column, axis=1)
        y = df_enc[y_column]

        # two-step use of ElasticNetCV()-then-ElasticNet()
        preclass = ElasticNetCV(
            selection="random", random_state=self.seed, n_jobs=self.n_jobs
        )
        preclass.fit(x_train, y)
        # grab the l1_ratio and run again
        self.trained = ElasticNet(
            selection="random", random_state=self.seed, alpha=preclass.alpha_
        )
        self.trained.fit(x_train, y)

        # save to cache
        self.save_trained_model(
            df, verbose=verbose, description="Preprocessed with " + str(preprocessor)
        )

    @BEARTYPE
    def predict(self, x_test: pd.DataFrame, y_true: pd.Series) -> float:
        """
        Using an already trained model, predicts values in y_true. The Mean Absolute Pecent Error result is returned.

        Note: this function assumes that df is pre-processed as is typical for pymasq internal usage.

        Parameters
        ----------
        x_test : pd.Dataframe
            A data frame containing the data to train the model including a label column for binary classification
        y_true : pd.Series
            A pandas series contains contiunous values as ground truth

        Returns
        -------
        float:
            The MAPE score, a value between 0.0 and 1.0

        """
        assert self.trained is not None

        y_predict = self.trained.predict(x_test)
        return mape(y_true=y_true, y_pred=y_predict)


#########################
class LogisticRegressionClassifier(ModelingBase):
    @BEARTYPE
    def __init__(
        self,
        cache_location: Optional[str] = None,
        modeling_task: Optional[str] = None,
    ):
        """
        Create an instance of sklearn's Logistic Regression classifier.

        Parameters
        ----------
        cache_location : str, optional (Default: None)
            A string for the directory path to the cache for the current data.
        modeling_task : str, optional (Default: None) {"binary", "multi_class", "regression"}
            A string defining the type of modeling task
        """
        super().__init__(
            name="logreg", cache_location=cache_location, modeling_task=modeling_task
        )

    @BEARTYPE
    def train(
        self,
        df: pd.DataFrame,
        y_column: str,
        preprocessor: Optional[Type[PreprocessorBase]],
        retrain: bool = False,
        verbose: int = 0,
        **kwargs,
    ):
        """
        Train based on the given dataframe. Model is saved to the cache location.

        Parameters
        ----------
        df : pd.Dataframe
            A data frame containing the data to train the model including a label column for regression
        y_column : str
            The column name in the data frame that contains binary labels of ground truth
        preprocessor : PreprocessorBase  (Default: None)
            A child of PreprocessorBase class indicating what preprocessor to use. Options are:
            - pymasq.preprocessing.EmbeddingsEncoder
            - pymasq.preprocessing.LabelEncoder_pm
            - None (i.e., the data is already pre-processed)
        retrain: bool (Default: False)
            Ignore cached results and retrain
        verbose: int (Default 0)
            0 for no verbose output
            1 for verbose output
        """
        # super().train() loads cache if it exists, and returns pre-processed df for training
        df_enc = super().train(
            df, y_column, preprocessor, retrain, verbose, seed=self.seed
        )

        # we found a cache file
        if self.trained is not None:
            return

        # no cache found, we need to train.
        if verbose > 0:
            print("Training Logistic Regression model ")
        x_train = df_enc.drop(y_column, axis=1)
        y = LabelEncoder().fit_transform(df_enc[y_column])

        self.trained = LogisticRegressionCV(
            random_state=self.seed,
            n_jobs=self.n_jobs,
            solver=DEFAULT_LOGISITIC_REGRESSION_SOLVER,
        )
        self.trained.fit(x_train, y)

        # save to cache
        self.save_trained_model(
            df, verbose=verbose, description="Preprocessed with " + str(preprocessor)
        )

    @BEARTYPE
    def predict(self, x_test: pd.DataFrame, y_true: pd.Series) -> float:
        """
        Using an already trained model, predicts values in y_true. The Area Under the Curve result is returned.

        Note: this function assumes that df is pre-processed as is typical for pymasq internal usage.

        Parameters
        ----------
        x_test : pd.Dataframe
            A data frame containing the data to train the model including a label column for binary classification
        y_true : List of strings
            A list of strings containing labels as ground truth

        Returns
        -------
        float:
            The AUC score, a value between 0.0 and 1.0

        """
        assert self.trained is not None

        if pd.Series(y_true).nunique() == 2:
            y_predict = self.trained.predict(x_test)
            return roc_auc_score(y_true=y_true.tolist(), y_score=y_predict)
        else:
            y_predict = self.trained.predict_proba(x_test)
            return roc_auc_score(
                y_true=y_true.tolist(), y_score=y_predict[:, 1:], multi_class="ovr"
            )


#########################
class RFClassifier(ModelingBase):
    @BEARTYPE
    def __init__(
        self, cache_location: Optional[str] = None, modeling_task: Optional[str] = None
    ):
        """
        Create an instance of sklearn's Random Forest classifier.

        Parameters
        ----------
        cache_location : str, optional (Default: None)
            A string for the directory path to the cache for the current data.
        modeling_task : str, optional (Default: None) {"binary", "multi_class", "regression"}
            A string defining the type of modeling task
        """
        super().__init__(
            name="rfclass", cache_location=cache_location, modeling_task=modeling_task
        )

    @BEARTYPE
    def train(
        self,
        df: pd.DataFrame,
        y_column: str,
        preprocessor: Optional[Type[PreprocessorBase]],
        retrain: bool = False,
        verbose: int = 0,
        **kwargs,
    ):
        """
        Train based on the given dataframe. Model is saved to the cache location.

        Parameters
        ----------
        df : pd.Dataframe
            A data frame containing the data to train the model including a label column for regression
        y_column : str
            The column name in the data frame that contains binary labels of ground truth
        preprocessor : PreprocessorBase  (Default: None)
            A child of PreprocessorBase class indicating what preprocessor to use. Options are:
            - pymasq.preprocessing.EmbeddingsEncoder
            - pymasq.preprocessing.LabelEncoder_pm
            - None (i.e., the data is already pre-processed)
        retrain: bool (Default: False)
            Ignore cached results and retrain
        verbose: int (Default 0)
            0 for no verbose output
            1 for verbose output
        """
        # super().train() loads cache if it exists, and returns pre-processed df for training
        df_enc = super().train(
            df, y_column, preprocessor, retrain, verbose, seed=self.seed
        )

        # we found a cache file
        if self.trained is not None:
            return

        # no cache found, we need to train.
        if verbose > 0:
            print("Training Logistic Regression model ")
        x_train = df_enc.drop(y_column, axis=1)
        y = LabelEncoder().fit_transform(df_enc[y_column])

        self.trained = RandomForestClassifier(
            n_jobs=self.n_jobs, random_state=self.seed
        )
        self.trained.fit(x_train, y)

        # save to cache
        self.save_trained_model(
            df, verbose=verbose, description="Preprocessed with " + str(preprocessor)
        )

    @BEARTYPE
    def predict(self, x_test: pd.DataFrame, y_true: pd.Series) -> float:
        """
        Using an already trained model, predicts values in y_true. The Area Under the Curve result is returned.

        Note: this function assumes that df is pre-processed as is typical for pymasq internal usage.

        Parameters
        ----------
        x_test : pd.Dataframe
            A data frame containing the data to train the model including a label column for binary classification
        y_true : List of strings
            A list of strings containing labels as ground truth

        Returns
        -------
        float:
            The AUC score, a value between 0.0 and 1.0

        """
        assert self.trained is not None

        if pd.Series(y_true).nunique() == 2:
            y_predict = self.trained.predict(x_test)
            return roc_auc_score(y_true=y_true.tolist(), y_score=y_predict)
        else:
            y_predict = self.trained.predict_proba(x_test)
            return roc_auc_score(
                y_true=y_true.tolist(), y_score=y_predict, multi_class="ovr"
            )


#########################
class RFRegressor(ModelingBase):
    @BEARTYPE
    def __init__(self, cache_location: Optional[str] = None):
        """
        Create an instance of sklearn's Random Forest regressor.

        Parameters
        ----------
        cache_location : str, optional (Default: None)
            A string for the directory path to the cache for the current data.

        """
        super().__init__(name="rfreg", cache_location=cache_location)

    @BEARTYPE
    def train(
        self,
        df: pd.DataFrame,
        y_column: str,
        preprocessor: Optional[Type[PreprocessorBase]],
        retrain: bool = False,
        verbose: int = 0,
    ):
        """
        Train sklearn's Random Forest regressor. The trained model is stored in RFRegressor.cache_location.

        Parameters
        ----------
        df : pd.Dataframe
            A data frame containing the data to train the model including a numeric column for regression
        y_column : str
            The column name in the data frame that contains numeric values of ground truth
        preprocessor : PreprocessorBase
            A child of PreprocessorBase class indicating what preprocessor to use. Options are:
            - pymasq.preprocessing.EmbeddingsEncoder
            - pymasq.preprocessing.LabelEncoder_pm
            - None (i.e., the data is already pre-processed)
        retrain: bool (Default: False)
            Ignore cached results and retrain
        verbose: in (Default 0)
            0 for no verbose output
            1 for verbose logging
        """
        # this call to super() loads the cache if it exists, and then returns a  pre-processed version for training.
        # some preprocessors are not expecting a seed kw argument.
        df_enc = super().train(
            df, y_column, preprocessor, retrain, verbose, seed=self.seed
        )

        # we found it in the cache
        if self.trained is not None:
            return

        # We didn't: train the model and cache it.
        if verbose > 0:
            print("Training LarsCV model ")
        x_train = df_enc.drop(y_column, axis=1)
        y = LabelEncoder().fit_transform(df_enc[y_column])
        self.trained = RandomForestRegressor(n_jobs=self.n_jobs, random_state=self.seed)
        self.trained.fit(x_train, y)

        # save to cache
        self.save_trained_model(
            df, verbose=verbose, description="Preprocessed with " + str(preprocessor)
        )

    @BEARTYPE
    def predict(self, x_test: pd.DataFrame, y_true: pd.Series) -> float:
        """
        Using an already trained model, predicts values in y_true. The Mean Absolute Pecent Error result is returned.

        Note: this function assumes that df is pre-processed as is typical for pymasq internal usage.

        Parameters
        ----------
        x_test : pd.Dataframe
            A data frame containing the data to train the model including a numeric column for regression
        y_true : pd.Series
            A pandas series contains continuous values as ground truth

        Returns
        -------
        float:
            The MAPE score, a value between 0.0 and 1.0

        """
        assert self.trained is not None

        y_predict = self.trained.predict(x_test)
        return mape(y_true=y_true, y_pred=y_predict)


#########################
class TpotClassifier(ModelingBase):
    @BEARTYPE
    def __init__(
        self, cache_location: Optional[str] = None, modeling_task: Optional[str] = None
    ):
        """
        Create an instance of sklearn's TPOT classifier.

        Parameters
        ----------
        cache_location : str, optional (Default: None)
            A string for the directory path to the cache for the current data.
        modeling_task : str, optional (Default: None) {"binary", "multi_class", "regression"}
            A string defining the type of modeling task
        """
        super().__init__(
            name="tpotclass", cache_location=cache_location, modeling_task=modeling_task
        )

    @BEARTYPE
    def train(
        self,
        df: pd.DataFrame,
        y_column: str,
        preprocessor: Optional[Type[PreprocessorBase]],
        scoring: Optional[Any] = "f1",
        generations: int = 25,
        population_size: int = 25,
        cv: int = 5,
        retrain: bool = False,
        use_dask: bool = False,
        verbose: int = 0,
    ):
        """
        Uses TPOT auto-ML to find the best model pipeline that can distinguish between the binary labels in y_column of
        the data frame. Saves the trained model in a cache directory.

        Parameters
        ----------
        df : pd.Dataframe
            A data frame containing the data to train the model including a label column for binary classification

        y_column : str
            A string of column name in the data frame that contains binary labels of ground truth

        preprocessor : PreprocessorBase
            A child of PreprocessorBase class indicating what preprocessor to use. Options are:
            - pymasq.preprocessing.EmbeddingsEncoder
            - pymasq.preprocessing.LabelEncoder_pm
            - None (i.e., the data is already pre-processed)

        scoring : string or callable, optional (Default: 'f1')
            Function used to evaluate the quality of a given pipeline for the
            problem. By default, accuracy is used for classification problems and
            mean squared error (MAPE) for regression problems.

            Offers the same options as sklearn.model_selection.cross_val_score as well as
            a built-in score 'balanced_accuracy'.

            Classification metrics:
            {'accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',
            'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
            'precision', 'precision_macro', 'precision_micro', 'precision_samples',
            'precision_weighted', 'recall', 'recall_macro', 'recall_micro',
            'recall_samples', 'recall_weighted', 'roc_auc'}

        generations : int or None, optional (Default: 25)
            Number of iterations to the run pipeline optimization process.
            It must be a positive number or None. If None, the parameter
            max_time_mins must be defined as the runtime limit.
            Generally, TPOT will work better when you give it more generations (and
            therefore time) to optimize the pipeline. TPOT will evaluate
            POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.

        population_size : int, optional (Default: 25)
            Number of individuals to retain in the GP population every generation.
            Generally, TPOT will work better when you give it more individuals
            (and therefore time) to optimize the pipeline. TPOT will evaluate
            POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.

        cv : int or cross-validation generator, optional (Default: 5)
            If CV is a number, then it is the number of folds to evaluate each
            pipeline over in k-fold cross-validation during the TPOT optimization
            process. If it is an object then it is an object to be used as a
            cross-validation generator.

        retrain : boolean, optional (Default: False)
            Re-runs and saves over existing TPOT model for the given file path.

        use_dask : boolean, optional (Default: False)
            Whether to use Dask-ML's pipeline optimiziations. This avoid re-fitting
            the same estimator on the same split of data multiple times. It
            will also provide more detailed diagnostics when using Dask's
            distributed scheduler.

        verbose: integer, optional (Default: 0)
            How much information TPOT communicates while it's running.

            Possible inputs are:
            0, TPOT will print nothing,
            1, TPOT will print minimal information,
            2, TPOT will print more information and provide a progress bar, or
            3, TPOT will print everything and provide a progress bar.
        """
        # loads cache, if it exists, and returns pre-processed df
        df_enc = super().train(
            df, y_column, preprocessor, retrain, verbose, seed=self.seed
        )

        # we found a cache file.
        if self.trained is not None:
            return

        # No cache available, we need to train
        if verbose > 0:
            print(f"{type(self).__name__} Training new model.")

        tpot = TPOTClassifier(
            generations=int(generations),
            population_size=int(population_size),
            scoring=scoring,
            cv=int(cv),
            random_state=self.seed,
            n_jobs=self.n_jobs,
            use_dask=use_dask,
            warm_start=True,
            verbosity=verbose,
        )

        x_train = df_enc.drop(y_column, axis=1)
        y = LabelEncoder().fit_transform(df_enc[y_column])

        tpot.fit(features=x_train, target=y)
        self.trained = tpot.fitted_pipeline_

        # save to cache
        self.save_trained_model(
            df, verbose=verbose, description="Preprocessed with " + str(preprocessor)
        )

    @BEARTYPE
    def predict(self, x_test: pd.DataFrame, y_true: pd.Series) -> float:
        """
        Using an already trained model, predicts values in y_true. The Area Under the Curve result is returned.

        Note: this function assumes that df is pre-processed as is typical for pymasq internal usage.

        Parameters
        ----------
        x_test : pd.Dataframe
            A data frame containing the data to train the model including a label column for binary classification
        y_true : List of strings
            A list of strings containing labels as ground truth

        Returns
        -------
        float:
            The AUC score, a value between 0.0 and 1.0


        """
        assert self.trained is not None

        # deal with fact that Tpot doesn't always provide all these functions
        Y_predict_prob_array = None
        for fn in ["predict_proba", "predict_prob", "decision_function"]:
            try:
                Y_predict_prob_array = getattr(self.trained, fn)(x_test)
                if fn in ["predict_proba", "predict_prob"]:
                    Y_predict_prob_array = Y_predict_prob_array[:, 1]
            except:
                continue
        if Y_predict_prob_array is None:
            raise (f"No prediction method available for {self.trained}")

        if pd.Series(y_true).nunique() == 2:
            return roc_auc_score(y_true=y_true.tolist(), y_score=Y_predict_prob_array)
        else:
            return roc_auc_score(
                y_true=y_true.tolist(),
                y_score=Y_predict_prob_array[:, 1:],
                multi_class="ovr",
            )


#########################
class TpotRegressor(ModelingBase):
    @BEARTYPE
    def __init__(self, cache_location: Optional[str] = None):
        """
        Create an instance of TPOT regressor.

        Parameters
        ----------
        cache_location : str, optional (Default: None)
            A string for the directory path to the cache for the current data.

        """
        super().__init__(name="tpotreg", cache_location=cache_location)

    @BEARTYPE
    def train(
        self,
        df: pd.DataFrame,
        y_column: str,
        preprocessor: Optional[Type[PreprocessorBase]],
        scoring: Optional[Any] = "f1",
        generations: int = 25,
        population_size: int = 25,
        cv: int = 5,
        retrain: bool = False,
        use_dask: bool = False,
        verbose: int = 0,
    ):
        """
        Uses TPOT auto-ML to find the best model pipeline that can perform regressoin on the y_column of
        the data frame. Saves the trained model in a cache directory.

        Parameters
        ----------
        df : pd.Dataframe
            A data frame containing the data to train the model including a numeric column for regression

        y_column : str
            A string of column name in the data frame that contains numeric values of ground truth

        preprocessor : PreprocessorBase
            A child of PreprocessorBase class indicating what preprocessor to use. Options are:
            - pymasq.preprocessing.EmbeddingsEncoder
            - pymasq.preprocessing.LabelEncoder_pm
            - None (i.e., the data is already pre-processed)

        scoring : string or callable, optional (Default: 'f1')
            Function used to evaluate the quality of a given pipeline for the
            problem. By default, accuracy is used for classification problems and
            mean squared error (MAPE) for regression problems.

            Offers the same options as sklearn.model_selection.cross_val_score as well as
            a built-in score 'balanced_accuracy'.

            Classification metrics:
            {'accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',
            'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
            'precision', 'precision_macro', 'precision_micro', 'precision_samples',
            'precision_weighted', 'recall', 'recall_macro', 'recall_micro',
            'recall_samples', 'recall_weighted', 'roc_auc'}

        generations : int or None, optional (Default: 25)
            Number of iterations to the run pipeline optimization process.
            It must be a positive number or None. If None, the parameter
            max_time_mins must be defined as the runtime limit.
            Generally, TPOT will work better when you give it more generations (and
            therefore time) to optimize the pipeline. TPOT will evaluate
            POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.

        population_size : int, optional (Default: 25)
            Number of individuals to retain in the GP population every generation.
            Generally, TPOT will work better when you give it more individuals
            (and therefore time) to optimize the pipeline. TPOT will evaluate
            POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.

        cv : int or cross-validation generator, optional (Default: 5)
            If CV is a number, then it is the number of folds to evaluate each
            pipeline over in k-fold cross-validation during the TPOT optimization
            process. If it is an object then it is an object to be used as a
            cross-validation generator.

        retrain : boolean, optional (Default: False)
            Re-runs and saves over existing TPOT model for the given file path.

        use_dask : boolean, optional (Default: False)
            Whether to use Dask-ML's pipeline optimiziations. This avoid re-fitting
            the same estimator on the same split of data multiple times. It
            will also provide more detailed diagnostics when using Dask's
            distributed scheduler.

        verbose: integer, optional (Default: 0)
            How much information TPOT communicates while it's running.

            Possible inputs are:
            0, TPOT will print nothing,
            1, TPOT will print minimal information,
            2, TPOT will print more information and provide a progress bar, or
            3, TPOT will print everything and provide a progress bar.
        """
        # loads cache, if it exists, and returns pre-processed df
        df_enc = super().train(
            df, y_column, preprocessor, retrain, verbose, seed=self.seed
        )

        # we found a cache file.
        if self.trained is not None:
            return

        # No cache available, we need to train
        if verbose > 0:
            print(f"{type(self).__name__} Training new model.")

        tpot = TPOTRegressor(
            generations=int(generations),
            population_size=int(population_size),
            scoring=scoring,
            cv=int(cv),
            random_state=self.seed,
            n_jobs=self.n_jobs,
            use_dask=use_dask,
            warm_start=True,
            verbosity=verbose,
        )

        x_train = df_enc.drop(y_column, axis=1)
        y = df_enc[y_column]

        tpot.fit(features=x_train, target=y)
        self.trained = tpot.fitted_pipeline_

        # save to cache
        self.save_trained_model(
            df, verbose=verbose, description="Preprocessed with " + str(preprocessor)
        )

    @BEARTYPE
    def predict(self, x_test: pd.DataFrame, y_true: pd.Series) -> float:
        """
        Using an already trained model, predicts values in y_true. The Mean Absolute Pecent Error result is returned.

        Note: this function assumes that df is pre-processed as is typical for pymasq internal usage.

        Parameters
        ----------
        x_test : pd.Dataframe
            A data frame containing the data to train the model including a numeric column for regression
        y_true : pd.Series
            A pandas series contains numeric values as ground truth

        Returns
        -------
        float:
            The MAPE score, a value between 0.0 and 1.0

        """
        assert self.trained is not None

        # deal with fact that Tpot doesn't always provide all these functions
        Y_predict_prob_array = None
        for fn in ["predict_proba", "predict_prob", "decision_function"]:
            try:
                Y_predict_prob_array = getattr(self.trained, fn)(x_test)
                if fn in ["predict_proba", "predict_prob"]:
                    Y_predict_prob_array = Y_predict_prob_array[:, 1]
            except:
                continue
        if Y_predict_prob_array is None:
            raise (f"No prediction method available for {self.trained}")

        return mape(y_true=y_true, y_score=Y_predict_prob_array)


# For translation from text to callable functions
model_fn = {
    "encv": ElasticNetCvRegressor,
    "rfreg": RFRegressor,
    "tpotreg": TpotRegressor,
    "larscv": LarsCvRegressor,
    "logreg": LogisticRegressionClassifier,
    "rfclass": RFClassifier,
    "tpotclass": TpotClassifier,
}
