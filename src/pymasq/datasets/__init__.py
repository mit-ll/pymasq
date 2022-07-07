"""
The :mod:`pymasq.datasets` module includes utilities to load tabular datasets.
"""

from ._base import load_data, load_census, load_loan, load_prestige, load_bank_attrition_rates
from .data_generator import gen_geom_seq, gen_bin_df, gen_num_df
from .utils import rand_cat_change


__all__ = [
    "load_data",
    "load_census",
    "load_loan",
    "load_prestige",
    "load_bank_attrition_rates",
    "rand_cat_change",
    "gen_geom_seq",
    "gen_bin_df",
    "gen_num_df",
]
