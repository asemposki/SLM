import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

colors = [
    "r",
    "b",
    "g",
    "k",
    "purple",
    "m",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "orange",
    "y",
]

__all__ = ["plot_eigs", "plot_dmd", "plot_S"]

base_PATH = os.path.join(os.path.dirname(__file__), "..")
PLOTS_PATH = f"{base_PATH}/Plots"

# plt.style.use("science")
mpl.rcParams["text.usetex"] = True
mpl.rcParams["axes.linewidth"] = 1.5


def _enforce_ratio(goal_ratio, supx, infx, supy, infy):
    """
    Code used from pyDMD package to plot the eigenvalues.
    Computes the right value of `supx,infx,supy,infy` to obtain the desired
    ratio in :func:`plot_eigs`. Ratio is defined as
    ::
        dx = supx - infx
        dy = supy - infy
        max(dx,dy) / min(dx,dy)

    :param float goal_ratio: the desired ratio.
    :param float supx: the old value of `supx`, to be adjusted.
    :param float infx: the old value of `infx`, to be adjusted.
    :param float supy: the old value of `supy`, to be adjusted.
    :param float infy: the old value of `infy`, to be adjusted.
    :return tuple: a tuple which contains the updated values of
        `supx,infx,supy,infy` in this order.
    """

    dx = supx - infx
    if dx == 0:
        dx = 1.0e-16
    dy = supy - infy
    if dy == 0:
        dy = 1.0e-16
    ratio = max(dx, dy) / min(dx, dy)

    if ratio >= goal_ratio:
        if dx < dy:
            goal_size = dy / goal_ratio

            supx += (goal_size - dx) / 2
            infx -= (goal_size - dx) / 2
        elif dy < dx:
            goal_size = dx / goal_ratio

            supy += (goal_size - dy) / 2
            infy -= (goal_size - dy) / 2

    return (supx, infx, supy, infy)


def _plot_limits(eigs, narrow_view):
    if narrow_view:
        supx = max(eigs.real) + 0.05
        infx = min(eigs.real) - 0.05

        supy = max(eigs.imag) + 0.05
        infy = min(eigs.imag) - 0.05

        return _enforce_ratio(8, supx, infx, supy, infy)
    return np.max(np.ceil(np.absolute(eigs)))


def plot_eigs(
    eigs,
    show_axes=True,
    show_unit_circle=True,
    figsize=(8, 8),
    title="",
    narrow_view=False,
    dpi=None,
    filename=None,
):
    """
    Plot the eigenvalues.
    :param bool show_axes: if True, the axes will be shown in the plot.
    :param bool show_unit_circle: if True, a unit circle centered at the origin will be shown.
    :param tuple(int,int) figsize: tuple defining the figure size in inches. Default is (8, 8).
    :param str title: title of the plot.
    :param narrow_view bool: if True, the plot will show only the smallest rectangular area containing
        all the eigenvalues, with a padding of 0.05. Not compatible with `show_axes=True`.
    :param dpi int: If not None, passed to ``plt.figure``.
    :param str filename: if specified, the plot is saved at `filename`.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title(title)

    (points,) = ax.plot(
        eigs.real, eigs.imag, "bo", label="Eigenvalues"
    )  # Comma unpacks to Line2D object

    if narrow_view:
        supx, infx, supy, infy = _plot_limits(eigs, narrow_view)
        ax.set_xlim((infx, supx))
        ax.set_ylim((infy, supy))

        if show_axes:
            endx = min(supx, 1.0)
            ax.annotate(
                "",
                xy=(endx, 0.0),
                xytext=(max(infx, -1.0), 0.0),
                arrowprops=dict(arrowstyle="->" if endx == 1.0 else "-"),
            )
            endy = min(supy, 1.0)
            ax.annotate(
                "",
                xy=(0.0, endy),
                xytext=(0.0, max(infy, -1.0)),
                arrowprops=dict(arrowstyle="->" if endy == 1.0 else "-"),
            )
    else:
        limit = _plot_limits(eigs, narrow_view)
        ax.set_xlim((-limit, limit))
        ax.set_ylim((-limit, limit))

        if show_axes:
            ax.annotate(
                "",
                xy=(max(limit * 0.8, 1.0), 0.0),
                xytext=(min(-limit * 0.8, -1.0), 0.0),
                arrowprops=dict(arrowstyle="->"),
            )
            ax.annotate(
                "",
                xy=(0.0, max(limit * 0.8, 1.0)),
                xytext=(0.0, min(-limit * 0.8, -1.0)),
                arrowprops=dict(arrowstyle="->"),
            )

    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")

    # Unit circle
    if show_unit_circle:
        unit_circle = plt.Circle(
            (0.0, 0.0),
            1.0,
            color="green",
            fill=False,
            linestyle="--",
            label="Unit circle",
        )
        ax.add_artist(unit_circle)

    ax.grid(True, linestyle="-.")

    # Legend handling
    if show_unit_circle:
        ax.legend([points, unit_circle], ["Eigenvalues", "Unit circle"], loc="best")
    else:
        ax.legend([points], ["Eigenvalues"], loc="best")

    ax.set_aspect("equal")

    if filename:
        plt.savefig(f"{PLOTS_PATH}/{filename}")
    else:
        plt.show()


def plot_dmd(t, X, Xdmd, fileNames, ylabels, fileName=None):
    """Makes plots for the dmds"""
    linT = np.arange(len(X[0]))
    namesList = fileName.strip(".dat").split("_")
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = plt.axes()

    for i in range(len(Xdmd)):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
        ax.plot(t, np.exp(Xdmd[i].real), label="DMD")
        ax.plot(linT, np.exp(X[i]), ".", label="data")

        ax.set_xlabel(r"Index", fontsize=22)
        ax.set_ylabel(ylabels[i], fontsize=22)
        # ax.set_title(, fontsize=18)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=18,
            labelright=False,
            direction="in",
            right=True,
            top=True,
            size=8,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=18,
            labelright=False,
            direction="in",
            right=True,
            top=True,
            size=4,
        )
        plt.legend(loc=f"best", prop={"size": 10})
        title_parts = [
            rf"$p_{i}$ = {namesList[2+i]}" for i in range(len(namesList[2:]))
        ]
        titleName = ", ".join(title_parts)
        if len(namesList) > 2:
            ax.set_title(titleName)
        plt.savefig(f"{PLOTS_PATH}/{fileNames[i]}")
        plt.close()


def plot_dmd_rad(X, Xdmd, fileNames, ylabels, fileName=None):
    """Makes plots for the dmds"""
    linT = np.arange(len(X[0]))
    namesList = fileName.strip(".dat").split("_")
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = plt.axes()

    for i in range(len(Xdmd) - 1):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
        ax.plot(np.exp(Xdmd[0].real), np.exp(Xdmd[i + 1].real), label="DMD", zorder=2)
        ax.plot(np.exp(X[0]), np.exp(X[i + 1]), ".", label="data", zorder=1)
        
        # ax.set_xlim(0.9 * min(np.exp(X[0])), 1.1 * max(np.exp(X[0])))
        # ax.set_ylim(0.9 * min(np.exp(X[i + 1])), 1.1 * max(np.exp(X[i + 1])))
        if i == 0:
            plt.yscale("log")

        ax.set_xlabel(r"Radius [km]", fontsize=22)
        ax.set_ylabel(ylabels[i], fontsize=22)
        # ax.set_title(, fontsize=18)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=18,
            labelright=False,
            direction="in",
            right=True,
            top=True,
            size=8,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=18,
            labelright=False,
            direction="in",
            right=True,
            top=True,
            size=4,
        )
        plt.legend(loc="best", prop={"size": 10})
        title_parts = [f"p{i} = {namesList[2+i]}" for i in range(len(namesList[2:]))]
        titleName = ", ".join(title_parts)

        if len(namesList) > 2:
            ax.set_title(titleName)
        # ax.set_title(f"{namesList[1]}")
        plt.savefig(f"{PLOTS_PATH}/{fileNames[i]}")
        plt.close()


def plot_S(S):
    # Plot the S values:
    plt.plot(S / S[0], marker=".", markersize=10)
    plt.yscale("log")
    plt.xlabel("Index")
    plt.ylabel("S values")
    plt.savefig(f"{PLOTS_PATH}/Svalues.pdf")
    plt.cla()


def plot_parametric(Xdmd, X, name, tidal=False):
    """Makes plots for the dmds"""
    # newT = np.delete(t, -1)
    names = name.split("_")
    print("Names", names)
    print("Shape", Xdmd.shape, X.shape)
    if tidal:
        n = 2
    else:
        n = 1
    for i in range(n):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
        color = colors[i]
        ax.plot(
            np.exp(Xdmd[0].real),
            np.exp(Xdmd[i + 2].real),
            color=color,
            label="DMD",
        )
        plt.plot(X[0], X[i + 2], ".", color=color, label="-data")
        # ax.set_ylim([0, 3])
        # ax.set_xlim([0, 30])
        plt.suptitle([f"p_{i} = {names[1+i]}" for i in range(len(names[1:]))])
        ax.set_xlabel(r"Radius (km)", fontsize=22)
        ax.set_ylabel(r"Mass $(M_{\odot})$", fontsize=22)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=18,
            labelright=False,
            direction="in",
            right=True,
            top=True,
            size=8,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=18,
            labelright=False,
            direction="in",
            right=True,
            top=True,
            size=4,
        )
        plt.legend(loc="upper right", prop={"size": 10})
        if i == 0:
            plt.savefig("MRpredict_" + "_".join(names) + ".png")
        else:
            plt.savefig("Tidalpredict_" + "_".join(names) + ".png")
        plt.close()
        plt.cla()


if __name__ == "__main__":
    pass
