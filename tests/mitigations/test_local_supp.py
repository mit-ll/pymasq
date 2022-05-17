import pandas as pd

from pymasq import config

config.FORMATTING_ON_OUTPUT = True
config.FORMATTING_IGNORE_DTYPES = True

from pymasq.mitigations.utils import _as_dataframe
from pymasq.mitigations import local_supp

dataset = {
    "ZipCode": [13055, 13056, 13057, 13058, 13056, 13055, 13058, 14859, 14859, 13057],
    "Age": [45, 46, 40, 50, 46, 45, 50, 54, 54, 40],
    "Nationality": [
        "USA",
        "USA",
        "Russian",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "Chilean",
    ],
    "Disease": [
        "Heart Disease",
        "Viral Infection",
        "Cancer",
        "Cancer",
        "Cancer",
        "Heart Disease",
        "Viral Infection",
        "Viral Infection",
        "Heart Disease",
        "Viral Infection",
    ],
}

dataset = pd.DataFrame(dataset)


def testing_ls_zero_threshold():
    df = _as_dataframe(dataset)
    supp_col = local_supp(df, "Nationality", "Disease", to_val="NA", risk_threshold=0.0)
    expected_result = pd.Series(["NA"] * 10)

    assert all(supp_col["Nationality"].eq(expected_result)), "This should be true"


def testing_ls_one_threshold():
    df = _as_dataframe(dataset)
    supp_col = local_supp(df, "Nationality", "Disease", to_val="NA", risk_threshold=1.0)

    expected_result = dataset["Nationality"]

    assert all(supp_col["Nationality"].eq(expected_result)), "This should be true"


def testing_ls_ignore_cols():
    df = _as_dataframe(dataset)
    supp_col = local_supp(df, "Nationality", "Disease", to_val="NA", risk_threshold=0.6)

    expected_result = pd.Series(["USA"] * 2 + ["NA"] + ["USA"] * 6 + ["NA"])

    assert all(supp_col["Nationality"].eq(expected_result)), "This should be true"


def testing_ls_ignore_threshold():
    df = _as_dataframe(dataset)
    supp_col = local_supp(
        df, "Nationality", "Disease", to_val="NA", risk_threshold=0.6, max_unique=4
    )

    expected_result = pd.Series(["USA"] * 2 + ["NA"] + ["USA"] * 6 + ["NA"])

    assert all(supp_col["Nationality"].eq(expected_result)), "This should be true"
