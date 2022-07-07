from os.path import join
from typing import Optional

import pandas as pd

from pymasq import ROOT_DIR, BEARTYPE


DATA_DIR = join(ROOT_DIR, "datasets", "data")


@BEARTYPE
def load_data(fname: str, fpath: Optional[str] = None):
    """Load and return the data from fpath/fname.

    Parameters
    ----------
    fname : {"census.csv", "loan.csv", "prestige.csv"
             "shuffled.csv", "y_star.csv", "y_star1.csv"}
        Name of csv file to be loaded from `fpath`.

    fpath : string, optional
        File path where `fname` is located.
        The module's data directory will be used if omitted.

    Returns
    -------
    pandas.DataFrame

    """
    if fpath is None:
        fpath = DATA_DIR

    return pd.read_csv(join(fpath, fname))


def load_census():
    """Load and return the Adult Census Income dataset.

    The adult census income dataset contains information regarding individuals from the 1994 Census database.
    The `sex` feature can be used to evaluate the effectiveness Statistical Disclosure Control methods.

    ==============   ==============
    Samples total             32562
    Dimensionality               15
    ==============   ==============

    Returns
    -------
    pandas.DataFrame

    The dataset was obtained from:
    http://archive.ics.uci.edu/ml/datasets/Adult

    """
    return load_data("census.csv")


def load_loan():
    """Load and return the Loan dataset.

    The loan dataset contains mock information regarding home loans issued by the Dream Housing Finance company.
    The `gender`, `married`, `education`, `self_employed`, and `loan_status` are all prime feature candidates
    that can be used to evaluate the effectiveness Statistical Disclosure Control methods.

    ==============   ==============
    Samples total               614
    Dimensionality               13
    ==============   ==============

    Returns
    -------
    pandas.DataFrame

    The dataset was obtained from:
    https://www.kaggle.com/burak3ergun/loan-data-set
    """
    return load_data("loan.csv")

def load_bank_attrition_rates():
    """Load and return the Bank Attrition Rates dataset.

    A manager at the bank is disturbed with more and more customers leaving their credit card services. 
    They would really appreciate if one could predict for them who is gonna get churned so 
    they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction.

    ==============   ==============
    Samples total            10,127
    Dimensionality               20
    ==============   ==============

    Returns
    -------
    pandas.DataFrame

    The dataset was obtained from:
    https://www.kaggle.com/sakshigoyal7/credit-card-customers

    The `CLIENTNUM` and `Naive_Bayes_Classifier_Attrition_Flag_*` columns have been removed.
    """
    return load_data("bank-attrition-rates.csv")


def load_prestige():
    """Load and return the Prestige Of Canadian Occupations dataset.

    ==============   ==============
    Samples total               102
    Dimensionality              6
    ==============   ==============

    This data frame contains the following columns:

    education
    Average education of occupational incumbents, years, in 1971.

    income
    Average income of incumbents, dollars, in 1971.

    women
    Percentage of incumbents who are women.

    prestige
    Pineo-Porter prestige score for occupation, from a social survey conducted
    in the mid-1960s.

    census
    Canadian Census occupational code.

    type
    Type of occupation. A factor with levels (note: out of order):
    bc, Blue Collar; prof, Professional, Managerial, and Technical;
    wc, White Collar.

    Returns
    -------
    pandas.DataFrame

    The dataset was obtained from:
    https://www.rdocumentation.org/packages/carData/versions/3.0-4/topics/Prestige
    """
    return load_data("prestige.csv")


def load_shuffled():
    """Load and return the shuffled Prestige Of Canadian Occupations dataset.
    This data is the result of running R sdcMicro shuffle on the Prestige
    dataset for test and validation of shuffle.py.

    ==============   ==============
    Samples total               102
    Dimensionality              3
    ==============   ==============

    This data frame contains the following columns:

    Index
    Type of occupation. A factor with levels (note: out of order):
    bc, Blue Collar; prof, Professional, Managerial, and Technical;
    wc, White Collar.

    education
    Average education of occupational incumbents, years, in 1971.

    income
    Average income of incumbents, dollars, in 1971.

    Returns
    -------
    pandas.DataFrame

    The dataset was originally obtained from:
    https://www.rdocumentation.org/packages/carData/versions/3.0-4/topics/Prestige
    """
    return load_data("shuffled.csv")


def load_y_star():
    """Load and return the shuffled Prestige Of Canadian Occupations dataset.
    This data is the result of running R sdcMicro shuffle without random noise
    on the Prestige dataset for test and validation of shuffle.py.

    ==============   ==============
    Samples total               102
    Dimensionality              3
    ==============   ==============

    This data frame contains the following columns:

    Index
    Type of occupation. A factor with levels (note: out of order):
    bc, Blue Collar; prof, Professional, Managerial, and Technical;
    wc, White Collar.

    income
    Average income of incumbents, dollars, in 1971.

    education
    Average education of occupational incumbents, years, in 1971.

    Returns
    -------
    pandas.DataFrame

    The dataset was originally obtained from:
    https://www.rdocumentation.org/packages/carData/versions/3.0-4/topics/Prestige
    """
    return load_data("y_star.csv")


def load_y_star1():
    """Load and return the shuffled Prestige Of Canadian Occupations dataset.
    This data is the result of running R sdcMicro shuffle on the Prestige
    dataset for test and validation of shuffle.py.

    ==============   ==============
    Samples total               102
    Dimensionality              2
    ==============   ==============

    This data frame contains the following columns:

    income
    Average income of incumbents, dollars, in 1971.

    education
    Average education of occupational incumbents, years, in 1971.

    Returns
    -------
    pandas.DataFrame

    The dataset was originally obtained from:
    https://www.rdocumentation.org/packages/carData/versions/3.0-4/topics/Prestige
    """
    return load_data("y_star1.csv")
