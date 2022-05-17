import numpy as np
import pandas as pd
import string
import random
from typing import List, Union

from .utils import rand_cat_change

from pymasq import BEARTYPE


@BEARTYPE
def gen_geom_seq(start: float = 0.5, n: int = 6, rate: float = 2.0) -> List[float]:
    """
    Builds a list of floats that geometrically increase for n times from a starting value

    Parameters
    ----------
    start : float, optional
        The starting value for generating a doubled values
        (Default: .5)

    n : int, optional
        Number of time that the value will be doubled
        (Default: 6)

    rate : float, optional
        Rate of increases or decreases in the sequence
        (Default: 2.0)

    Returns
    -------
    x_s : List[float]
        List of geometric sequence
    """
    x_s: List[float] = [start]
    for _ in range(1, n):
        x_s.append(x_s[-1] * rate)
    return x_s


@BEARTYPE
def gen_bin_df(n: int = 1000):
    """
    Generates a dataframe of eight (8) binary columns where the first column
    is considered the target variable with half 0's and half 1's, and then
    the next six columns have progressively more (0% - 50% by 10%) of the values
    from the traget column flipped that that they are less correlated, and the
    last column is all 0's.

    This dataset can be used for distinguishing the importance of a column to
    predicting the target column, such that each column is pregressively less
    correlated.

    Parameters
    ----------
    n : int, optional
        Number of rows in the returned dataframe
        (Default: 1000)

    Returns
    -------
    pd.DataFrame
        Dataframe containing increasingly unimportant binary variables
    """
    if n % 2 == 0:
        df_dict = {"Label": [0] * int(n / 2) + [1] * int(n / 2)}
    else:
        df_dict = {"Label": [0] * int(n / 2) + [1] * (int(n / 2) + 1)}
    for i in range(0, 50, 10):
        df_dict["Perc_" + str(i)] = rand_cat_change(np.array(df_dict["Label"]), i / 100)
    df_dict["One_Cat"] = [0] * n
    return pd.DataFrame(df_dict)


@BEARTYPE
def gen_num_df(n: int = 1000, seed: int = 1234) -> pd.DataFrame:
    """
    Generates a dataframe of seven (7) numeric columns where the first column
    is considered the target variable with half 0's and half 1's, and then
    the next six columns have samples from two normal distributions (means 5 and 10,
    corresponding to 0 and 1 respectively) with progressively larger standard
    deviations.  This will generate columns with data that increasingly overlaps
    between the two classes and therefore decreases the ability to distinguish
    betweenthe two classes of 0 and 1.  This should correspond to decrease in
    variable importance.

    Parameters
    ----------
    n : int, optional
        Number of rows in the returned dataframe
        (Default: 1000)

    seed : int, optional
        Number to set for random seed used in the gauss function
        (Default: 1234)

    Returns
    -------
    pd.DataFrame
        Dataframe containing increasingly unimportant numeric variables
    """
    random.seed(seed)
    if n % 2 == 0:
        df_dict = {"Label": [0] * int(n / 2) + [1] * int(n / 2)}
        for i in gen_geom_seq():
            df_dict["STDEV_" + str(i)] = [
                random.gauss(5, i) for _ in range(int(n / 2))
            ] + [random.gauss(10, i) for _ in range(int(n / 2))]
    else:
        df_dict = {"Label": [0] * int(n / 2) + [1] * (int(n / 2) + 1)}
        for i in gen_geom_seq():
            df_dict["STDEV_" + str(i)] = [
                random.gauss(5, i) for _ in range(int(n / 2))
            ] + [random.gauss(10, i) for _ in range(int(n / 2) + 1)]
    return pd.DataFrame(df_dict)


@BEARTYPE
def _l_div_sensitive_gen(l: int, n: int) -> List:
    """
    Generates the sensitive variable for generate_l_diverse_table for each equivalence class
    Parameters
    ----------
    l : int
        The specified diversity that the equivalence class needs to be
    n : int
        The size of the equivalence class (i.e. the lenght of the list returned)
    Returns
    -------
    List[int]
        List of integer values for the sensitive column
    """

    unique_entries = np.random.choice(range(n), l)
    while len(unique_entries) != len(set(unique_entries)):
        unique_entries = np.random.choice(range(n), l)

    non_unique = np.random.choice(unique_entries, n - l)
    return list(unique_entries) + list(non_unique)


@BEARTYPE
def generate_l_diverse_table(
    l: Union[int, List[int]],
    num_col: int = 5,
    num_q_blocks: int = 5,
    q_block_sizes: Union[int, List[int]] = 5,
) -> pd.DataFrame:
    """
    Used for testing l-diversity. Creates a data set that is l-diverse for given l.
    Parameters
    ----------
    l : Union[int, List[int]]
        The specified diversity that the data set needs to be TODO: need to expand this to allow float l parameters for entropy
    num_col : int, optional
        The number of columns (in addition to the sensitive column) the data set should have
        (default: 5)
    num_q_blocks : int, optional
        The number of q_blocks (equivalence classes) the data set should have
        (default: 5)
    q_block_sizes : Union[int, List[int]], optional
        Specified sizes of the q_blocks; if an int q_block_sizes will be the size of all q_blocks, otherwise must be a list of length num_q_blocks
    Returns
    -------
    pd.DataFrame
        A dataframe that meets the above requirements for testing of the l-diversity metric
    """

    col_names = {"col_{}".format(i): [] for i in range(num_col)}
    col_con = {
        "col_{}".format(i): list(string.ascii_lowercase)[i] for i in range(num_col)
    }
    col_names["sensitive"] = []

    q_block_sizes = (
        [q_block_sizes] * num_q_blocks
        if isinstance(q_block_sizes, int)
        else q_block_sizes
    )
    l = [l] * num_q_blocks if not isinstance(l, list) else l

    for n in range(num_q_blocks):
        senn = _l_div_sensitive_gen(l[n], q_block_sizes[n])
        col_names["sensitive"] += senn
        for cn in col_names:
            if cn != "sensitive":
                col_names[cn] += [col_con[cn] * (n + 1)] * q_block_sizes[n]

    return pd.DataFrame(col_names)


@BEARTYPE
def generate_t_close_table(
    sensitive_col: List,
    num_col: int = 3,
    num_q_blocks: int = 3,
    q_block_sizes: Union[int, List[int]] = 3,
) -> pd.DataFrame:
    """
    Used for testing t-closeness. Creates a data set that is t-close for given t.
    Parameters
    ----------
    sensitive_col : List[int]
        List to be used as the sensitive column in the dataset
    num_col : int, optional
        The number of columns (in addition to the sensitive column) the data set should have
        (default: 3)
    num_q_blocks : int, optional
        The number of q_blocks (equivalence classes) the data set should have
        (default: 3)
    q_block_sizes : Union[int, List[int]], optional
        Specified sizes of the q_blocks; if an int q_block_sizes will be the size of all q_blocks, otherwise must be a list of length num_q_blocks
        (default: 3)
    Returns
    -------
    pd.DataFrame
        A dataframe that meets the above requirements for testing of the l-diversity metric
    """

    col_names = {"col_{}".format(i): [] for i in range(num_col)}
    col_con = {
        "col_{}".format(i): list(string.ascii_lowercase)[i] for i in range(num_col)
    }
    col_names["sensitive"] = sensitive_col

    q_block_sizes = (
        [q_block_sizes] * num_q_blocks
        if isinstance(q_block_sizes, int)
        else q_block_sizes
    )

    for n in range(num_q_blocks):
        for cn in col_names:
            if cn != "sensitive":
                col_names[cn] += [col_con[cn] * (n + 1)] * q_block_sizes[n]

    return pd.DataFrame(col_names)
