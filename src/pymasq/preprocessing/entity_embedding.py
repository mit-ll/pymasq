import hashlib
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from tensorflow.random import set_seed as tf_set_seed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

from pymasq.utils import cache
import pymasq.config as cfg


def embed_cache_fn(column: pd.Series, cache_location: Path) -> Path:
    """
    Determine the filename for a cached instance of a embed_entity. This result
    will be a prefix for cache.save(), which will add an hmac suffix

    Parameters
    ----------
    column: pd.Series
        The column of a data frame that the embedding was based on
    cache_location: Path
        The full path to the cache

    Returns
    -------
    Path:
        The full path and filename prefix of the cached file.
    """
    uid = hashlib.md5(pd.util.hash_pandas_object(column).values).hexdigest()
    return cache_location / (uid + ".embedding")


def embed_entities(
    target_df: Union[pd.DataFrame, pd.Series],
    categorical_df: pd.DataFrame,
    epochs: int = 1000,
    num_embedding_components: int = 6,
    cache_location: Optional[Union[str, Path]] = None,
    verbose: Union[bool, int] = 0,
    learning_rate: float = 0.01,
    seed: int = None,
    retrain=False,
) -> Dict[str, np.array]:
    """
    Performs Keras embedding on target_df given the categorical data provided by categorical_df
    and the information provided by the optional parameters. Writes the embedded dictionary to a
    pickle file in cache_dir and returns the embedded dictionary.

    Parameters
    ----------
    target_df : pd.DataFrame or pd.Series
        Data frame containing the target variable or y
    categorical_df : pd.DataFrame
        Data frame containing categorical data to be embedded
    epochs : int, optional (Default: 1000)
        The maximum number of training epochs to run
    num_embedding_components : int, optional (Default: 6)
        The number of components that each categorical variable will be mapped to
    cache_location : str or Path, optional (Default: pymasq.config.cache_location)
        The directory path as a string to save out the entity embeddings.
        If passed None, the embeddings are not saved.
    verbose : int {0, 1, 2}, optional (Default: 0)
        Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
    learning_rate : float, optional (Default: 0.01)
        Learning rate for the neural network back propagation
    retrain: bool (Default: False)
        Set to True to force training despite cached value.

    Returns
    -------
    Dict
        The embeddings.
    The embeddings are also saved to the given cache_location.

    See Also
    --------
    Embedding

    """

    cache_location = (
        cfg.CACHE_LOCATION if cache_location is None else Path(cache_location)
    )
    cache_location.mkdir(parents=True, exist_ok=True)
    seed = cfg.DEFAULT_SEED if seed is None else seed

    if verbose > 0:
        print(f"Tensor flow seed set to {seed}.")
    tf_set_seed(seed)

    embed_dict = {}

    early_stopping = EarlyStopping(
        monitor="loss",
        min_delta=0.2 * learning_rate,
        patience=10,
        mode="min",
        restore_best_weights=True,
    )

    for column in categorical_df.columns:
        if cache_location is not None:
            filename = embed_cache_fn(categorical_df[column], cache_location)
            if not retrain:
                # ignore description
                embed_dict[column], _ = cache.load_cache(filename)
                if verbose > 1:
                    print("\t Cache file found and loaded for column", column)
                # returns none if a file was found but hmac didn't match
                if embed_dict[column] is not None:
                    continue

        if verbose > 1:
            print("\tembed_entities: No cache available for ", column)

        # Converts categories represented by integers to strings so that the
        # label encoder will work and the classes can be determined later
        categorical_df[column] = categorical_df[column].astype(str)
        le = LabelEncoder()
        X_train = le.fit_transform(categorical_df[column])

        model = Sequential()
        model.add(
            Embedding(
                input_dim=len(le.classes_),
                output_dim=num_embedding_components,
                input_shape=(1,),
            )
        )
        model.add(Flatten())

        metrics_array = {}
        loss_array = {}
        ys = []
        if isinstance(target_df, pd.Series):
            target_df = pd.DataFrame(target_df, columns=[target_df.name])
        for col in target_df.columns:
            name = f"output_{col}"
            new_y = LabelEncoder().fit_transform(target_df[col].values)
            ys.append(new_y)
            n_cats = len(target_df[col].unique())
            if n_cats > 2:
                model.add(Dense(n_cats, activation="softmax", name=name))
                metrics_array[name] = "sparse_categorical_accuracy"
                loss_array[name] = "sparse_categorical_crossentropy"
            else:
                # A Dense Layer is created for each output
                model.add(Dense(1, activation="sigmoid", name=name))
                metrics_array[name] = "binary_accuracy"
                loss_array[name] = "binary_crossentropy"

        y = np.array(ys).T
        sgd = SGD(learning_rate=learning_rate)
        model.compile(optimizer=sgd, loss=loss_array, metrics=metrics_array)
        model.fit(
            X_train,
            y,
            epochs=epochs,
            verbose=0,
            callbacks=[early_stopping],
            use_multiprocessing=True,
        )
        embedding_basis = model.layers[0].get_weights()[0]
        embed_dict[column] = embedding_basis

        cache.save(
            embed_dict[column],
            str(cache_location),
            filename,
            description=f"Embedding of {column}.",
        )

    return embed_dict
