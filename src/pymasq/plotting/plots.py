from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_results(
    df: pd.DataFrame,
    risk_col: str = "scores.risk_reduction",
    util_col: str = "scores.utility",
) -> go.Figure:
    """
    Returns a plotly graph object based on the provided data frame and columns

    Parameters
    ----------
    df : pd.DataFrame
        The pandas data frame to provide data for the plot
    risk_col : str, optional
        The data frame column to be used as the Y axis (Default: 'scores.risk_reduction')
    util_col : str, optional
        The data frame column to be used as the X axis (Default: 'scores.utility')

    Returns
    -------
    go.Figure
        The plotly graph object generated from the provided data frame and optional parameters

    """
    fig = px.scatter(
        df,
        x=util_col,
        y=risk_col,
        color="val_cat",
        height=600,
        width=600,
        hovername="Method: {}".format(df["log"]),
        color_discrete_map={"good": "darkgreen", "fair": "gold", "poor": "red"},
    )
    fig.add
    return fig


def plot_auc(plot_data: Dict, rrscore: float):
    """
    Plots the data provided in `plot_data` and the reduced risk score the original data in black
    and the modified data in red

    Parameters
    ----------
    plot_data : Dict
        A dictionary of data to be plotted with keys: ['orig_fpr', 'orig_tpr', 'mod_fpr', 'mod_tpr']
    rrscore : float
        The reduced risk score

    Returns
    -------
    None
        Plots the provided data
    """
    plt.title("Receiver Operating Characteristic")
    plt.plot(
        plot_data["orig_fpr"],
        plot_data["orig_tpr"],
        color="black",
        label="Original Data ROC Curve",
    )
    plt.plot(
        plot_data["mod_fpr"],
        plot_data["mod_tpr"],
        color="blue",
        marker="o",
        linestyle="dashed",
        label="Modified Data ROC Curve",
    )
    plt.text(0.5, 0.25, s="Reduced Risk Score = %0.2f" % rrscore)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()
