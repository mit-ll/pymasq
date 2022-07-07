from time import time
from typing import Tuple, List, Union, Optional
import numpy as np
import pandas as pd
from bpemb import BPEmb
from pathlib import Path
from sklearn.decomposition import TruncatedSVD, PCA, IncrementalPCA, KernelPCA, FastICA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder as skLabelEncoder

from pymasq.errors import InputError
import pymasq.config as cfg
from .entity_embedding import embed_entities
from pymasq.preprocessing._base import PreprocessorBase
from pymasq.metrics import utils
from pymasq import BEARTYPE

# This file contains two children of PreprocessorBase
#  1. LabelEncoder_pm
#  2. EmbeddingsEncoder

#################

REDUCTION_METHODS = {
    "pca": PCA,
    "trucated": TruncatedSVD,
    "incremental": IncrementalPCA,
    "kernel": KernelPCA,
    "fast": FastICA,
}


class LabelEncoder_pm(PreprocessorBase):
    """
    This class manages an instance of sklearn's LabelEncoder.
    Encodes categorical data only, as integers.
    """

    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    @BEARTYPE
    def encode(df: Union[pd.Series, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Uses sklearn's LableEncoder to encode a single data frame. This method encodes categorical columns only,
        relabelling all as integers. Categorical columns are identifed as those that are not dtype "number".

        If you need encode two dataframes in a consistent manner, then use encode_both(), and do not call this
        method twice.

        **kwargs are ignored currently.

        Parameter
        ---------
        df: pdf.DataFrame:
            The data frame to encode.

        Returns
        -------
        pd.DataFrame
            df preprocessed so that categorical data is relabeled as integers.
        """
        le = skLabelEncoder()
        col_order = df.columns.tolist()
        cat_cols = df.select_dtypes(exclude=["number"])
        # cast everything to string in case we have a mix of floats and string,
        # otherwise LabelEncoder will choke/die.
        # This should never happen, but does in our pytests, so who knows.
        cat_cols = cat_cols.apply(
            lambda col: le.fit_transform(col.astype(str)), axis=0, result_type="expand"
        )
        num_cols = df.select_dtypes(include=["number"])

        if cat_cols.empty:
            both = num_cols
        else:
            both = pd.concat(
                [cat_cols.apply(le.fit_transform), num_cols], join="outer", axis=1
            )
        return both[col_order]

    @staticmethod
    @BEARTYPE
    def encode_both(
        df_A: pd.DataFrame, df_B: pd.DataFrame, **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Takes two dataframes and uses sklearn's LabelEncoder on categorical columns only to relabel
        all categories as integers. The labels are consistent across the two data frames;
        in other words, if df A has a category that df B does not, the integers in B reflect
        that. Does not modify the dataframes that are passed in, returns a copy.

        Categorical columns are identifed as those that are not dtype "number".

        This method requires that the two data frames have the same column names.

        **kwargs are ignored currently.

        Parameter
        ---------
        df_A: pdf.DataFrame:
            The data frame to encode.
        df_B: pdf.DataFrame:
            The data frame to encode.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]:
            pd.DataFrame
                df_A data frame now preprocessed so that categorical data is relabeled as integers.
            pd.DataFrame
                df_B data frame now preprocessed so that categorical data is relabeled as integers.

        Column order remains consistent with original dataframes. df_A and df_B are not modified.

        """
        le = skLabelEncoder()
        # make a copy
        df_a = df_A.copy()
        df_b = df_B.copy()
        if set(df_a.columns) != set(df_B.columns):
            raise InputError("df_A and df_B must have same columns")
        col_order = df_a.columns.tolist()

        # join together; mark each so we can separate again later
        # use a uniquely named column to do that
        class_col = utils.uniq_col_name(df_a)

        df_a[class_col] = 0
        df_b[class_col] = 1

        # append b to a; and then split out the categorical (non-numerical) columns
        cat_cols = df_a.append(df_b).select_dtypes(exclude=["number"])
        # cast everything to string in case we have a mix of floats and string,
        # otherwise LabelEncoder will choke/die.
        # This should never happen, but does in our pytests, so who knows.
        cat_cols = cat_cols.apply(
            lambda col: le.fit_transform(col.astype(str)), axis=0, result_type="expand"
        )

        # append b to a; and then split out the non-categorical (numerical) columns
        num_cols = df_a.append(df_b).select_dtypes(include=["number"])
        # concatenate, but relabel the cat_cols first
        if cat_cols.empty:
            both = num_cols
        else:
            both = pd.concat(
                [cat_cols.apply(le.fit_transform), num_cols], join="outer", axis=1
            )
        # split up again, and drop the extra column
        df_A_enc = both.loc[both[class_col] == 0].drop(class_col, axis=1)
        df_B_enc = both.loc[both[class_col] == 1].drop(class_col, axis=1)

        return df_A_enc[col_order], df_B_enc[col_order]


#################


@BEARTYPE
def _col_embeddings(embeddings: np.ndarray, col: pd.Series) -> pd.DataFrame:
    """
    Takes in a entity embedding array and column and returns the
    embeddings as columns in a dataframe. This is a helper function for
    EmbeddingsEncoder.

    Parameters
    ----------
    embeddings : array-like
        Array containing the basis vectors for a given column in a dataset

    col : pd.Series
        a given column from a dataset

    Returns
    -------
    pd.DataFrame
        Data frame containing the number of categorical entity embeddings
        from a given column
    """
    cat_dummies = pd.get_dummies(col)
    embedding_array = np.dot(cat_dummies, embeddings)
    embed_columns = [col.name + "_" + str(j) for j in range(embeddings.shape[1])]
    return pd.DataFrame(embedding_array, index=col.index, columns=embed_columns)


class EmbeddingsEncoder(PreprocessorBase):
    """
    The encoder works on both categorical and numerical column types.
    """

    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def sentence_bpe_vectors(
        sents: Union[pd.Series, List[str]],
        num_embedding_components: int,
        reduct_method: str = "pca",
    ) -> np.ndarray:
        """
        Uses Byte Pair Embedding to vectorize english sentences

        Parameters
        ----------
        sents: array of strings
            A list of (a list of strings)

        num_embedding_components: int
            The dimention of the vector representing each sentences

        reduct_method: str {"pca", "truncated", "kernel", "incremental", "fast"}
            Type of dimensionality reduction used on embeddings

        Returns
        -------
        np.ndarray:
            The results as a 1-d ndarray of (floats) of length model.vector_size
        """
        vectorizer = BPEmb(lang=cfg.BPE_LANG, dim=cfg.BPE_DIM)
        embeddings = vectorizer.embed(sents).mean(axis=1)
        reduct = REDUCTION_METHODS[reduct_method](n_components=num_embedding_components)
        embeddings = reduct.fit_transform(embeddings)
        return embeddings

    @staticmethod
    @BEARTYPE
    def encode_both(
        df_A: pd.DataFrame,
        df_B: pd.DataFrame,
        sensitive_col: Optional[Union[List, str]] = None,
        seed: int = 1234,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encodes two dataframes in a consistent manner. The sensitive_col is pased to encode() and processed according to that
        method's rules.

        Parameters
        ----------
        df_A : pd.DataFrame
            data frame containing the binary label column and the other variables
            of interest
        df_B : pd.DataFrame
            data frame containing the binary label column and the other variables
            of interest
        sensitive_col : str or List[str] (Default: None)
            name(s) of the column that contains the label data
        seed: int (Default: 1234)
            seed passed to tensor flow

        Return
        -------
        Tuple:
            pd.DataFrame
                The encoded version of df_A
            pd.DataFrame
                The encoded version of df_B
        df_A and df_B are not modified.
        """

        if set(df_A.columns) != set(df_B.columns):
            raise InputError("df_A and df_B must have same columns.")

        # pick a column name that isn't in the dataset
        class_col = utils.uniq_col_name(df_A)
        # make one dataframe for pre-processing, otherwise preprocess_data won't be consistent
        orig_df_copy = df_A.copy()
        mod_df_copy = df_B.copy()
        orig_df_copy[class_col] = 0
        mod_df_copy[class_col] = 1
        comb_for_proprocessing = pd.concat(
            [orig_df_copy, mod_df_copy], axis=0, sort=False
        ).reset_index(drop=True)
        # encode
        df_proc = EmbeddingsEncoder.encode(
            df=comb_for_proprocessing,
            sensitive_col=sensitive_col,
            ignore_columns=class_col,
            normalize=False,
            seed=seed,
        )
        # split it up again.
        orig_df_proc = (
            df_proc.loc[df_proc[class_col] == 0]
            .drop([class_col], axis=1)
            .reset_index(drop=True)
        )
        mod_df_proc = (
            df_proc.loc[df_proc[class_col] == 1]
            .drop([class_col], axis=1)
            .reset_index(drop=True)
        )
        # return both
        return orig_df_proc, mod_df_proc

    @staticmethod
    @BEARTYPE
    def get_numerical_cols(
        df: pd.DataFrame,
        sensitive_col: Optional[Union[List, str]] = None,
        ignore_columns: Optional[Union[List, str]] = None,
        text_threshold: float = 0.75,
    ) -> List[str]:
        """
        This method returns the list of numerical_cols as calculated by encode(). Needed for other code in pymasq. (e.g,. kve())

        Parameters
        ----------
        df : pd.DataFrame
            data frame containing the binary label column and the other variables
            of interest

        sensitive_col : str or List[str]
            name(s) of the column that contains the label data

        ignore_columns: str or List[str]
            name(s) of the columns that should not be encoded

        text_threshold : float, Optional
            percent of the rows in a categorical column that can contain unique,
            if there are more unique values then the column will be treated as a
            test column
            (Default: 0.75, 75% of the column can be unique
            before it is treated as a textual column versus a categorical)
        Returns
        ----------
        List[str]
            List of names of numerical columns in the dataframe
        """
        if ignore_columns is None:
            ignore_columns = []

        (
            _,
            numerical_columns,
            _,
            binary_cat_columns,
            binary_columns,
            _,
        ) = EmbeddingsEncoder._organize_columns(
            df, sensitive_col, ignore_columns, text_threshold
        )
        # Adds binary columns to numeric columns because these columns are not expanded
        numerical_columns.extend(binary_cat_columns)
        numerical_columns.extend(binary_columns)
        return numerical_columns

    @staticmethod
    @BEARTYPE
    def _organize_columns(
        df: pd.DataFrame,
        sensitive_col: Optional[Union[List, str]] = None,
        ignore_columns: Optional[Union[List, str]] = None,
        text_threshold: float = 0.75,
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        """
        Given a dataframe, return how the columns will be treated by encode().

        This private helper method is split off from encode() because it must also be called from get_numerical_cols().

        Return
        -------
        Returns in this order, each a list:
            - dropped_cols
            - numerical_columns
            - categorical_columns
            - binary_cat_columns
            - binary_columns
            - textual_columns
        """
        dropped_cols = []

        if sensitive_col:
            if isinstance(sensitive_col, str):
                sensitive_col = [sensitive_col]
            dropped_cols.extend(sensitive_col)

        if ignore_columns:
            if isinstance(ignore_columns, str):
                ignore_columns = [ignore_columns]
            dropped_cols.extend(ignore_columns)

        if sensitive_col or ignore_columns:
            input_data = df.drop(dropped_cols, 1).copy()
        else:
            input_data = df.copy()

        numerical_columns = list(
            input_data.select_dtypes(include=[np.number]).columns.values
        )
        binary_columns = [
            col
            for col in numerical_columns
            if np.isin(input_data[col].dropna().unique(), [0, 1]).all()
        ]
        numerical_columns = [
            col
            for col in numerical_columns
            if not np.isin(input_data[col].dropna().unique(), [0, 1]).all()
        ]

        cat_columns = list(
            input_data.select_dtypes(include=["object", "category"]).columns.values
        )
        textual_columns = []
        categorical_columns = []
        binary_cat_columns = []

        for col in cat_columns:
            n_unique = input_data[col].nunique()
            if (
                n_unique > input_data.shape[0] * text_threshold
                or input_data[col].str.len().max() > 200
            ):
                textual_columns.append(col)
            else:
                if n_unique == 2:
                    binary_cat_columns.append(col)
                else:
                    categorical_columns.append(col)

        return (
            dropped_cols,
            numerical_columns,
            categorical_columns,
            binary_cat_columns,
            binary_columns,
            textual_columns,
        )

    @staticmethod
    @BEARTYPE
    def encode(
        df: pd.DataFrame,
        sensitive_col: Optional[Union[List, str]] = None,
        ignore_columns: Optional[Union[List, str]] = None,
        missing_value: Optional[Union[str, int]] = None,
        text_threshold: float = 0.75,
        num_embedding_components: int = 6,
        retrain: bool = False,
        normalize: bool = False,
        cache_location: Optional[Union[str, Path]] = None,
        verbose: int = 0,
        seed: int = 1234,
    ) -> pd.DataFrame:
        """
        Takes in a data frame and prepares the data for use in machine learning
        algorithms based on the data types in the columns.

        Parameters
        ----------
        df : pd.DataFrame
            data frame containing the binary label column and the other variables
            of interest

        sensitive_col : str or List[str] (Default: None)
            name(s) of the column that contains the label data

        ignore_columns: str or List[str] (Default: [])
            name(s) of the columns that should not be encoded

        missing_value : str or integer, optional (Default: None)
            denotes the code used in the data to indicate missing data


        text_threshold : float, optional (Default: 0.75)
            percent of the rows in a categorical column that can contain unique,
            if there are more unique values then the column will be treated as a
            test column.
            The default of 0.75 means that 75% of the column can be unique
            before it is treated as a textual column versus a categorical

        num_embedding_components: int (Default: 6)
            Number of components, or columns, that will be generated for each
            categorical column

        retrain : boolean, optional (Default: False)
            flag to determine if the entity embeddings should be retrained


        normalize : boolean, optional (Default: False)
            flag to determine if numeric data should be normalized


        cache_location : str, {"config", None}, optional (Default: pymasq.config.CACHE_LOCATION)
            The directory path as a string to save out the entity embeddings.
            If passed None, the embeddings will not be saved

        verbose : boolean, optional (Default: False)
            flag to determine if print statements should be active


        seed : int, Optional (Default: 1234)
            Seed passed to tensor flow

        Returns
        -------
        pd.DataFrame
            A data frame that has been preprocessed in the correct manner to be
            used in machine learning algorithms
        """
        if ignore_columns is None:
            ignore_columns = []

        if cache_location is None:
            cache_location = cfg.CACHE_LOCATION
        else:
            cache_location = Path(cache_location)

        if verbose > 0:
            print("Preprocessing Data...")
            start = time()

        cache_location.mkdir(parents=True, exist_ok=True)
        if verbose > 0:
            print("cache_location for preprocess is: " + str(cache_location))

        # Remove the sensitive column and other columns from consideration.
        # We'll add them back in later.
        # We have to keep these separate from sensitive_col or code breaks
        # y is going to hold our sensitive_col data
        # ignore_col_data is going to hold our ignored column data
        # input_data is going to hold all remaining columns.
        # The rest of the function operates on input_data

        if sensitive_col:
            y = df.loc[:, sensitive_col].copy()
            if y.isnull().values.any():
                raise ValueError("Sensitive Column contains NaN or infinity.")
        else:
            y = pd.Series(df.index).copy()
        (
            dropped_cols,
            numerical_columns,
            categorical_columns,
            binary_cat_columns,
            binary_columns,
            textual_columns,
        ) = EmbeddingsEncoder._organize_columns(
            df, sensitive_col, ignore_columns, text_threshold
        )
        ignore_col_data = None

        if verbose > 0:
            print("Splitting Data into Numerical and Categorical Data...")

        if sensitive_col or ignore_columns:
            input_data = df.drop(dropped_cols, 1).copy()
            ignore_col_data = df.loc[:, ignore_columns].copy()
        else:
            input_data = df.copy()

        # Impute missing data for binary data columns with most frequent value
        binary = input_data.loc[:, binary_columns]
        if binary_columns:
            if verbose > 0:
                print("Imputing Missing Binary Data...")
            simple_imputer = SimpleImputer(strategy="most_frequent")
            binary = pd.DataFrame(
                simple_imputer.fit_transform(input_data[binary_columns]),
                index=input_data[binary_columns].index,
                columns=binary_columns,
            )

        numerical_imputed_normalized = pd.DataFrame()
        if numerical_columns:
            if verbose > 0:
                print("Imputing Missing Numerical Data...")
            simple_imputer = SimpleImputer(strategy="mean")
            simple_imputer.fit(input_data[numerical_columns])
            numerical_imputed = pd.DataFrame(
                simple_imputer.transform(input_data[numerical_columns]),
                index=input_data[numerical_columns].index,
                columns=numerical_columns,
            )
            if missing_value:
                simple_imputer = SimpleImputer(
                    missing_values=missing_value, strategy="mean"
                )
                simple_imputer.fit(numerical_imputed)
                numerical_imputed = pd.DataFrame(
                    simple_imputer.transform(numerical_imputed),
                    index=numerical_imputed.index,
                    columns=numerical_columns,
                )
            if normalize:
                numerical_imputed_normalized = (
                    numerical_imputed - np.mean(numerical_imputed)
                ) / np.std(numerical_imputed)
            else:
                numerical_imputed_normalized = numerical_imputed

        categorical_embeddings = []
        if categorical_columns:
            if verbose > 0:
                print("Imputing Missing Categorical Data...")
            simple_imputer = SimpleImputer(fill_value="None", strategy="constant")
            simple_imputer.fit(input_data[categorical_columns])
            categorical_imputed = pd.DataFrame(
                simple_imputer.transform(input_data[categorical_columns]),
                index=input_data[categorical_columns].index,
                columns=categorical_columns,
            )
            if missing_value:
                simple_imputer = SimpleImputer(
                    missing_values=missing_value, fill_value="None", strategy="constant"
                )
                simple_imputer.fit(categorical_imputed)
                categorical_imputed = pd.DataFrame(
                    simple_imputer.transform(categorical_imputed),
                    index=categorical_imputed.index,
                    columns=numerical_columns,
                )
            if verbose > 0:
                print("Creating/Loading Categorical Data Embeddings...")

            new_embeddings = embed_entities(
                target_df=y,
                categorical_df=categorical_imputed[categorical_columns],
                cache_location=cache_location,
                verbose=verbose,
                seed=seed,
            )
            for col in categorical_columns:
                embed = new_embeddings[col]
                categorical_embeddings.append(
                    _col_embeddings(embed, categorical_imputed[col])
                )

        if binary_cat_columns:
            le = skLabelEncoder()
            for col in binary_cat_columns:
                binary.loc[:, col] = le.fit_transform(input_data[col])

        textual_embeddings = []
        if textual_columns:
            if verbose > 0:
                print("Imputing Missing Textual Data...")
            simple_imputer = SimpleImputer(
                missing_values="", fill_value="None", strategy="constant"
            )
            simple_imputer.fit(input_data[textual_columns])
            textual_imputed = pd.DataFrame(
                simple_imputer.transform(input_data[textual_columns]),
                index=input_data.index,
                columns=textual_columns,
            )
            if verbose > 0:
                print("Creating Textual Data Embeddings...")
            for col in textual_columns:
                if verbose > 0:
                    print("\t" + col)
                sents = textual_imputed[col].str.lower().str.replace("[!?:/]", " ")

                textual_embedding_array = EmbeddingsEncoder.sentence_bpe_vectors(
                    sents, num_embedding_components
                )

                textual_embedding_columns = [
                    col + "_" + str(j) for j in range(num_embedding_components)
                ]
                textual_embedding = pd.DataFrame(
                    textual_embedding_array,
                    index=input_data.index,
                    columns=textual_embedding_columns,
                )

                textual_embeddings.append(textual_embedding)

        if verbose > 0:
            print("Preprocessing took: {} seconds".format(round(time() - start, 2)))

        if sensitive_col:
            return pd.concat(
                [y, ignore_col_data, numerical_imputed_normalized, binary]
                + categorical_embeddings
                + textual_embeddings,
                1,
            )

        return pd.concat(
            [ignore_col_data, numerical_imputed_normalized, binary]
            + categorical_embeddings
            + textual_embeddings,
            1,
        )


# Translation from strings to function names
preprocessor_fn = {
    None: PreprocessorBase,
    "embeddings": EmbeddingsEncoder,
    "label_encode": LabelEncoder_pm,
}
