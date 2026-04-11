import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, NullLocator, MaxNLocator
from matplotlib.legend import Legend

import pandas as pd
import seaborn as sns

from typing import Dict, Tuple, List, TypeVar

# Here we define the types for the dictionary
T = TypeVar('T', plt.figure, plt.axes, Legend)
S = TypeVar('S', float, int)
FigureType = Dict[str, T]

sns.set_style("white")

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 5.0,
                                             "lines.markersize": 8,
                                             "lines.markeredgewidth": 10,
                                             })


def stacked_barplots(x: str, y: str, hue: str, data: pd.DataFrame, col: str = None, nb_cols: int = None,
                     ylabel: str = None, xlabel: str = None, figsize: Tuple[int, int] = (15, 8),
                     colors: List[str] = ["#b2df8a", "#1f78b4", "#a6cee3"],
                     bbox_to_anchor: Tuple[float, float] = (0.62, .4), adjust: Dict[str, float] = {'right': 0.9},
                     orderby: Tuple[int, List[S]] = None) -> FigureType:
    """
    Plots stacked bar plots where with x values in the x axis and y values in the y axis and colored by hue.
    The plots are organized by col and plotted in a grid of columns = min(nb_cols, unique(cols)).
    :param x: (str) name of the column of the pandas.Dataframe to be used as x asis
    :param y: (str) name of the column of the pandas.Dataframe to use as y axis
    :param hue: (str) name of the column of the pandas.Dataframe to be used to color the stacked bar plot
    :param data: (pandas.DataFrame) dataframe containing the data to be plotted
    :param col: (str) name of the column of the pandas.Dataframe to be used as index for the different plot
    :param nb_cols: (int) maximum number of columns of the plotting grid
    :param ylabel: (str) label for the y axis
    :param xlabel: (str) label for the x axis
    :param figsize: tuple containing the size of the figure
    :param colors: list containing the colors used in the plot
    :param bbox_to_anchor: positioning of the legend in the figure
    :param adjust: dictionary containing the parameters to position the subplots in the figure
    :param orderby: Tuple(order, values). 1st element 0 or 1. 0 orders from smaller to biguer, 1 from big to small. 2nd element List of values that will determine the order of the plots.
    :return Dictionary(figure, axes, legend)
    """
    if col is None:
        return stacked_barplot(x, y, hue, data, ylabel, xlabel,
                               figsize, colors, bbox_to_anchor, adjust)

    order = False
    if orderby is not None:
        order = True
        order_idx = np.argsort(orderby[1])
        if orderby[0]:
            # order by biggest element
            order_idx = list(reversed(order_idx))

    hue_values = data[hue].unique()
    col_values = data[col].unique()
    if order:
        col_values = col_values[order_idx]
    nb_col_values = col_values.shape[0]

    # Generate subplots organized by nb_cols
    if (nb_col_values < nb_cols):
        nb_rows = 1
    elif ((nb_cols % nb_col_values) != 0):
        nb_rows = 1 + (nb_col_values // nb_cols)
    else:
        nb_rows = nb_col_values // nb_cols
    fig, axes = plt.subplots(nb_rows, nb_cols if (nb_cols <= nb_col_values) else nb_col_values, figsize=figsize)

    # Turn off the subplots that don't contain any data
    if (nb_col_values % nb_cols) != 0:
        nb_missing_plots = nb_cols - (nb_col_values % nb_cols)
        if nb_rows > 1:
            for i in range(nb_missing_plots):
                axes[-1, -1 - i].axis('off')
        else:
            for i in range(nb_missing_plots):
                axes[-1 - i].axis('off')

    for i, value in enumerate(col_values):
        tmp = data[data[col] == value]

        s = [tmp[tmp[hue] == hue_values[-1]][y].values]
        for idx in reversed(range(len(hue_values) - 1)):
            s.append(s[idx - 1] + tmp[tmp[hue] == hue_values[idx]][y].values)

        x_values = tmp[tmp[hue] == hue_values[0]][x].values
        if (nb_rows > 1):
            index = (i // nb_cols, i % nb_cols)
        else:
            index = i % nb_cols
        # Plot stacked bars
        for idx in reversed(range(len(hue_values))):
            axes[index].bar(x=x_values, height=s[idx], color=colors[idx % len(colors)])
            axes[index].tick_params(axis="y", direction="in", pad=8)
            axes[index].tick_params(axis="y", which="minor", direction="in")
            axes[index].yaxis.set_minor_locator(AutoMinorLocator(5))
        # plot the axis names
        if not (i % nb_cols):
            axes[index].set_ylabel(ylabel)
        if (i // nb_cols) == (nb_col_values // nb_cols):
            axes[index].set_xlabel(xlabel)
        axes[index].set_title('{} {}'.format(col, i + 1))

    fig.subplots_adjust(**adjust)
    L = fig.legend(hue_values, title=hue, bbox_to_anchor=bbox_to_anchor, borderaxespad=0., frameon=False)

    return {'figure': fig, 'axes': axes.flatten()[:nb_col_values], 'legend': L}


def stacked_barplot(x: str, y: str, hue: str, data: pd.DataFrame, ylabel: str = None, xlabel: str = None,
                    figsize: Tuple[int, int] = (15, 8), colors: List[str] = ["#b2df8a", "#1f78b4", "#a6cee3"],
                    bbox_to_anchor: Tuple[float, float] = (0.62, .4),
                    adjust: Dict[str, float] = {'right': 0.9}) -> FigureType:
    """
    Plots stacked bar plots where with x values in the x axis and y values in the y axis and colored by hue.
    The plots are organized by col and plotted in a grid of columns = min(nb_cols, unique(cols)).
    :param x: (str) name of the column of the pandas.Dataframe to be used as x asis
    :param y: (str) name of the column of the pandas.Dataframe to use as y axis
    :param hue: (str) name of the column of the pandas.Dataframe to be used to color the stacked bar plot
    :param data: (pandas.DataFrame) dataframe containing the data to be plotted
    :param ylabel: (str) label for the y axis
    :param xlabel: (str) label for the x axis
    :param figsize: tuple containing the size of the figure
    :param colors: list containing the colors used in the plot
    :param bbox_to_anchor: positioning of the legend in the figure
    :param adjust: dictionary containing the parameters to position the subplots in the figure
    :return Dictionary(figure, axes, legend)
    """
    hue_values = data[hue].unique()
    fig, ax = plt.subplots(figsize=figsize)

    for i, value in enumerate(col_values):
        s = [data[data[hue] == hue_values[-1]][y].values]
        for idx in reversed(range(len(hue_values) - 1)):
            s.append(s[idx - 1] + data[data[hue] == hue_values[idx]][y].values)

        x_values = data[data[hue] == hue_values[0]][x].values
        # Plot stacked bars
        for idx in reversed(range(len(hue_values))):
            ax.bar(x=x_values, height=s[idx], color=colors[idx % len(colors)])
        ax.tick_params(axis="y", direction="in", pad=8)
        ax.tick_params(axis="y", which="minor", direction="in")
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    fig.subplots_adjust(**adjust)
    L = fig.legend(hue_values, title=hue, bbox_to_anchor=bbox_to_anchor, borderaxespad=0., frameon=False)

    return {'figure': fig, 'axes': axes.flatten()[:nb_col_values], 'legend': L}

# def calculate_pie(cluster, public_account, rounds):
#     """
#     Calculates the points of the pie plot
#     """
#     tmp = strategies_pa_freq[(strategies_pa_freq['clusters'] == cluster) & (strategies_pa_freq['rounds'] == rounds) & (strategies_pa_freq['public_account'] == public_account)]['freq_action'].values
#     # first define the ratios
#     r1 = tmp[0]
#     r2 = r1 + tmp[1]

#     x = [0] + np.cos(np.linspace(0, 2 * np.pi * r1, 1000)).tolist()
#     y = [0] + np.sin(np.linspace(0, 2 * np.pi * r1, 1000)).tolist()
#     xy1 = np.column_stack([x, y])
#     s1 = np.abs(xy1).max()

#     x = [0] + np.cos(np.linspace(2 * np.pi * r1, 2 * np.pi * r2, 1000)).tolist()
#     y = [0] + np.sin(np.linspace(2 * np.pi * r1, 2 * np.pi * r2, 1000)).tolist()
#     xy2 = np.column_stack([x, y])
#     s2 = np.abs(xy2).max()

#     x = [0] + np.cos(np.linspace(2 * np.pi * r2, 2 * np.pi, 1000)).tolist()
#     y = [0] + np.sin(np.linspace(2 * np.pi * r2, 2 * np.pi, 1000)).tolist()
#     xy3 = np.column_stack([x, y])
#     s3 = np.abs(xy3).max()
#     return xy1, s1, xy2, s2, xy3, s3


# def piepointPlot(x: str, y: str, hue: str, data: pd.DataFrame, col=None : str, nb_cols=None : int, ylabel=None : str, xlabel=None : str, figsize=(15, 8): Tuple[int, int], colors=["#b2df8a", "#1f78b4", "#a6cee3"]: List[str], bbox_to_anchor=(0.62, .4), adjust={'right': 0.9}) -> FigureType:
#     # define some sizes of the scatter marker
#     sizes = np.array([500, 500, 500])
#     possible_actions = [0, 2, 4]
#     nb_actions = len(possible_actions)
#     colors = ['#edf8b1', '#7fcdbb', '#2c7fb8']


#     fig = plt.figure(figsize=(15, 8))
#     axes = [fig.add_subplot(int("{}2{}".format(int(np.ceil(nb_clusters/2)), i+1))) for i in range(nb_clusters)]

#     for cluster in range(nb_clusters):
#         size_dataset = len(strategies_pa_freq[strategies_pa_freq['clusters'] == (cluster+1)]['rounds'])
#         rounds = strategies_pa_freq[strategies_pa_freq['clusters'] == (cluster+1)]['rounds'].values
#         public_accounts = strategies_pa_freq[strategies_pa_freq['clusters'] == (cluster+1)]['public_account'].values
#         for i in range(size_dataset):
#             # First we calculate the points
#             xy1, s1, xy2, s2, xy3, s3 = calculate_pie(cluster+1, public_accounts[i], rounds[i])
#             axes[cluster].scatter(rounds[i], public_accounts[i], marker=(xy1, 0),
#                    s=s1 ** 2 * sizes, facecolor=colors[0])
#             axes[cluster].scatter(rounds[i], public_accounts[i], marker=(xy2, 0),
#                    s=s2 ** 2 * sizes, facecolor=colors[1])
#             axes[cluster].scatter(rounds[i], public_accounts[i], marker=(xy3, 0),
#                    s=s3 ** 2 * sizes, facecolor=colors[2])
#     handles = []
#     for i in range(nb_actions):
#         handles.append(mpatches.Patch(color=colors[i], label=str(possible_actions[i])))
#     for i, ax in enumerate(axes):
#         ax.set_title('cluster {}'.format(i+1))
#         ax.set_xlabel('rounds')
#         ax.set_ylabel('Public Account')
#     plt.legend(handles=handles, bbox_to_anchor=(1.25, 1.), borderaxespad=0.)
#     sns.despine()