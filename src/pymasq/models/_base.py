from abc import abstractmethod
from joblib.parallel import DEFAULT_N_JOBS
import pandas as pd
import os
from typing import Type, Optional, List, Union
from pymasq.utils import cache

import pymasq.config as cfg
from pymasq.preprocessing._base import PreprocessorBase
from pymasq import BEARTYPE


class ModelingBase:
    """
    All children of this base class must have __init__(), train(), and auc_predict() methods and
    must call super() for each.

    load_trained_model() and save_trained_model() are helpers that should not need to be
    redefined for child classes.
    """

    @BEARTYPE
    def __init__(
        self,
        name: str,
        cache_location: Optional[str] = None,
        modeling_task: Optional[str] = None,
        seed: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        """
        This method should be called by all children as it loads cache files.

        Parameters
        ----------
        name: str
            The name of the child class. Used in logging statements and in cache filenames.
        cache_location : str, optional (Default: None)
            A string for the directory path to the cache for the current data.
            If None, the location is taken from pymasq.config.cfg.cache_location
        modeling_task : str, optional (Default: None) {"binary", "multi_class", "regression"}
            A string defining the type of modeling task
        seed: integer, optional (Default: None)
            If not None, then the seed passed to the preprocessor, else uses DEFAULT_SEED from config.py
        n_jobs: integer, optional (Default: None)
            If not None, then n_jobs used for trainings modes, else uses DEFAULT_N_JOBS from config.py

        """
        if cache_location is None:
            self.cache_location = cfg.CACHE_LOCATION
        else:
            self.cache_location = cache_location
        assert self.cache_location is not None

        if not os.path.exists(self.cache_location):
            os.mkdir(self.cache_location)

        assert name is not None
        self.name = name
        self.trained = None
        self.modeling_task = modeling_task

        self.seed = cfg.DEFAULT_SEED if seed is None else seed
        self.n_jobs = cfg.DEFAULT_N_JOBS if n_jobs is None else n_jobs

    @abstractmethod
    @BEARTYPE
    def train(
        self,
        df: pd.DataFrame,
        y_column: str,
        preprocessor: Optional[Type[PreprocessorBase]],
        retrain: bool = False,
        verbose: Union[bool, int] = 0,
        **kwargs,
    ) -> pd.DataFrame:
        """
        This method contains code needed by any child class: it checks for a cached trained model and loads it;
        it then returns the pre-processed version of df

        Parameters
        ----------
        df: pd.Dataframe
            A data frame containing the data to train the model including a label column for binary classification
        y_column : str
            A string of column name in the data frame that contains binary labels of ground truth
        preprocessor : PreprocessorBase
            A child of PreprocessorBase class indicating what preprocessor to use. Options are:
            - pymasq.preprocessing.EmbeddingsEncoder
            - pymasq.preprocessing.LabelEncoder_pm
            - None (i.e., the data is already pre-processed)
        retrain : boolean, optional (Default: False)
            Re-runs and saves over existing TPOT model for the given file path.
        verbose: integer, optional (Default: 0)
            0 disables verbose output
            1 (or higher) enables verbose output

        Returns
        ------
        pd.dataframe:
            The pre-processed version of the dataframe df.

        Note that if a cached instance is found (and retrain is not False), then self.trained is set by this method.
        """
        # if it's cached, load the trained model
        if not retrain:
            self.load_trained_model(df, verbose)  # sets self.trained from file
            if self.trained and verbose > 0:
                print(
                    f"{self.name}: loading trained model from cache. (Set retrain=True to ignore cache.)"
                )

        # If we don't need preprocessing, we are done
        if preprocessor is None:
            return df.copy()
        # Else, return preprocessed dataframe
        return preprocessor().encode(
            df=df,
            sensitive_col=y_column,
            cache_location=self.cache_location,
            **kwargs,
        )

    @abstractmethod
    @BEARTYPE
    def predict(self, x_test: pd.DataFrame, y_true: pd.Series):
        """
        Using an already trained model, predicts values in y_true. The evaluation result is returned.
        For classification tasks it is Area Under the ROC Curve (AUC), for regression it is Mean Absolute
        Percentage Error (MAPE).

        Note: this function assumes that df is pre-processed as is typical for pymasq internal usage.

        Parameters
        ----------
        x_test : pd.Dataframe
            A data frame containing the data to train the model including a label column for binary classification
        y_true : pd.Series
            A pandas series contains binary labels as ground truth

        Returns
        -------
        float:
            The evaluation score, a value between 0.0 and 1.0
        """
        pass

    @BEARTYPE
    def save_trained_model(
        self,
        df: pd.DataFrame,
        description: str = "",
        verbose: Union[int, bool] = 0,
    ):
        """
        Save a trained model to cache based on the data frame that was used to train it.

        Parameters
        ----------
        df: pd.DataFrame
            The dataframe that was used to train the model. This value will be used to find the trained model in cache when loaded.
        description: str  (Default: "")
            A description to add to the cached file. The first ten rows of the df are also saved as part of the description.
        verbose: int (Default 0)
            0 disable verbose logging
            1 enable verbose logging

        """
        assert self.cache_location is not None
        assert self.name is not None

        # determine filename prefix (hmac will be suffix)
        fn_prefix = cache.df_to_filename_prefix(
            df, suffix=self.name, path=self.cache_location
        )
        # descrption to be added to cache file
        # TODO: expose a flag to not store
        desc = f"{self.name}. Description: {description}\nFirst ten rows:\n{df[0:min(len(df),10)]}"

        # cache it
        filename = cache.save(
            data=self.trained,
            cache_location=self.cache_location,
            fn_prefix=fn_prefix,
            description=desc,
            verbose=verbose,
        )
        if verbose > 0:
            print(f"{self.name} model trained and saved to: {filename}")

    @BEARTYPE
    def load_trained_model(
        self,
        df: pd.DataFrame,
        verbose: Union[bool, int] = 0,
        ignore_hmac: bool = False,
    ):
        """
        Load the cached trained model for a given dataframe

        Parameters
        ----------
        df: pd.DataFrame
            The dataframe that was used to train the model. This value is used to find the trained model in cache.
        verbose: int (Default 0)
            0 disable verbose logging
            1 enable verbose logging

        """
        assert self.cache_location is not None
        assert self.name is not None

        # determine filename prefix (hmac will be suffix)
        filename_prefix = cache.df_to_filename_prefix(
            df, suffix=self.name, path=self.cache_location
        )
        # load it
        self.trained, _ = cache.load_cache(
            prefix_path=filename_prefix,
            verbose=verbose,
            ignore_hmac=ignore_hmac,
        )
