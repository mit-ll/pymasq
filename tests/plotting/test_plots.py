import pandas as pd
from pymasq.plotting import plot_results


def test_plot_results():
    df = pd.DataFrame({
        "scores.utility": [0.8, 0.9, 0.7],
        "scores.risk_reduction": [0.5, 0.6, 0.4],
        "val_cat": ["good", "fair", "poor"],
        "log": ["m1", "m2", "m3"]
    })
    fig = plot_results(df)
    assert fig is not None
