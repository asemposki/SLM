# In ~/Documents/Research/Alexandra/SLM/src/slmemulator/plotData.py

import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from .scripts import setup_rc_params
import numpy as np
import os

# from .config import get_paths # No longer needed for global PLOTS_PATH

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

__all__ = [
    "plot_eigs",
    "plot_slm",
    "plot_slm_rad",
    "plot_S",
    "plot_parametric",
    "plot_parametric_old",
]

# --- REMOVED GLOBAL PATH DEFINITION ---
# # Get paths for plots directory - REMOVED, now passed to functions
# paths = get_paths()
# PLOTS_PATH = paths["plots_dir"]
# # Ensure the plots directory exists - REMOVED, now created by individual plot functions
# os.makedirs(PLOTS_PATH, exist_ok=True)
# --- END REMOVAL ---

# plt.style.use("science")
mpl.rcParams["text.usetex"] = True
mpl.rcParams["axes.linewidth"] = 1.5
setup_rc_params()


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
    output_path=None,  # Renamed from 'filename' for clarity
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
    :param Path or str output_path: if specified, the plot is saved at this path.
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

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        plt.savefig(output_path)
        plt.close()  # Close the figure to free memory
    else:
        plt.show()


def plot_slm(
    t, X, Xdmd, fileNames_for_labels, ylabels, plots_output_base_path, base_name=None
):
    """
    Makes plots for the dmds in a time-series manner.

    Args:
        t (np.ndarray): Time vector or index for the x-axis.
        X (np.ndarray): Original data, typically (num_quantities, num_time_points).
        Xdmd (np.ndarray): Reconstructed DMD data, (num_quantities, num_time_points).
        fileNames_for_labels (list[str] or list[Path]): A list of strings or Path objects
                                                         used for generating plot titles and unique filenames.
                                                         Its length should correspond to len(Xdmd).
        ylabels (list[str]): A list of y-axis labels for each plot. Its length should match len(Xdmd).
        plots_output_base_path (Path or str): The base directory where plot files should be saved.
        base_name (str, optional): A base name to use for generated plot filenames,
                                   e.g., derived from the original data file.
    """
    plots_output_base_path = Path(plots_output_base_path)
    plots_output_base_path.mkdir(parents=True, exist_ok=True)

    linT = np.arange(
        len(X[0])
    )  # Using linT as an index if t is not provided for plot_slm

    # Use base_name for title parts if available, otherwise fallback to first label source
    namesList = []
    if base_name:
        namesList = Path(base_name).stem.split("_")
    elif fileNames_for_labels:
        namesList = Path(fileNames_for_labels[0]).stem.split("_")

    for i in range(len(Xdmd)):  # Loop for each quantity
        fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
        ax.plot(t, np.exp(Xdmd[i].real), label="DMD")  # Assuming Xdmd is log-scaled
        ax.plot(
            linT, np.exp(X[i]), ".", label="data"
        )  # Assuming X is log-scaled, adjust if not

        ax.set_xlabel(r"Index", fontsize=22)  # Or r"Time [units]" if t is actual time
        ax.set_ylabel(ylabels[i], fontsize=22)

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

        titleName = ""
        if len(namesList) > 2:  # Attempt to parse parameters from filename for title
            # This part needs to be adapted based on specific filename conventions
            title_parts = [
                rf"$p_{j}$ = {namesList[2+j]}" for j in range(len(namesList[2:]))
            ]
            titleName = ", ".join(title_parts)
            ax.set_title(titleName, fontsize=18)
        else:
            ax.set_title(f"Variable: {ylabels[i]}", fontsize=18)

        # Create a unique filename for the plot
        safe_ylabel = (
            ylabels[i]
            .replace(" ", "_")
            .replace("$", "")
            .replace("\\", "")
            .replace("{", "")
            .replace("}", "")
            .replace("^", "")
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        plot_file_name = (
            f"DMD_{Path(base_name).stem if base_name else 'plot'}_{safe_ylabel}.png"
        )

        plt.savefig(plots_output_base_path / plot_file_name)
        plt.close()


def plot_slm_rad(X, Xdmd, fileNames_for_labels, ylabels, plots_output_base_path):
    """
    Makes plots for the SLM DMD results compared to original data.

    Args:
        X (np.ndarray): The "real" or original data, typically (num_quantities, num_points).
                        Assumed X[0] is Radius, X[1:] are other quantities (e.g., Pressure, Energy Density).
                        Values should be log-scaled if np.exp is used for plotting.
        Xdmd (np.ndarray): The SLM-derived (reconstructed) data, (num_quantities, num_points).
                           Assumed Xdmd[0] is Radius, Xdmd[1:] are other quantities.
                           Values should be log-scaled if np.exp is used for plotting.
        fileNames_for_labels (list[str] or list[Path]): A list containing a single string or Path object.
                                                         This object is used to derive plot titles and base filenames,
                                                         as all plots from a single Xdmd set share the same EOS parameters.
        ylabels (list[str]): A list of y-axis labels for each plot. Its length should match len(Xdmd) - 1.
        plots_output_base_path (Path or str): The base directory where plot files should be saved.
    """
    # Ensure plots_output_base_path is a Path object and exists
    plots_output_base_path = Path(plots_output_base_path)
    plots_output_base_path.mkdir(parents=True, exist_ok=True)

    # The loop iterates (len(Xdmd) - 1) times, meaning it makes this many plots.
    # Each plot corresponds to a pair (X[0], X[i+1]) and (Xdmd[0], Xdmd[i+1])
    # fileNames_for_labels[i] and ylabels[i] should align with each plot.

    # Use the first (and presumably only) element of fileNames_for_labels
    # as the source for parameter information for the plot title and filename.
    # This assumes that all variables in a single Xdmd correspond to one EOS run.
    if not fileNames_for_labels:
        raise ValueError("fileNames_for_labels cannot be empty for plot_slm_rad.")

    base_label_source = Path(
        fileNames_for_labels[0]
    )  # Changed from fileNames_for_labels[i]

    for i in range(len(Xdmd) - 1):  # Loop for each quantity beyond Radius
        fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

        # Ensure values are positive before taking np.exp if they are log-scaled
        # Adding a small epsilon to avoid log(0) issues if Xdmd/X can contain 0 or negative
        ax.plot(np.exp(Xdmd[0].real), np.exp(Xdmd[i + 1].real), label="SLM")
        ax.plot(np.exp(X[0]), np.exp(X[i + 1]), ".", label="data")

        ax.set_xlabel(r"Radius [km]", fontsize=22)
        ax.set_ylabel(ylabels[i], fontsize=22)

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

        # Generate title and save filename based on base_label_source
        namesList = base_label_source.stem.split("_")  # Use base_label_source

        titleName = ""
        # Customize title based on your EOS naming conventions
        if namesList and namesList[0].lower() == "mseos" and len(namesList) >= 5:
            titleName = (
                f"MSEOS: Ls={float(namesList[1]):.4f}, Lv={float(namesList[2]):.3f}, "
                f"zeta={float(namesList[3]):.2e}, xi={float(namesList[4]):.2f}"
            )
        elif namesList and namesList[0].lower() == "qeos" and len(namesList) >= 3:
            titleName = f"Quarkyonia: Lambda={float(namesList[1]):.2f}, Kappa={float(namesList[2]):.2f}"
        else:  # For non-parametric or other unknown names
            titleName = f"EOS: {base_label_source.stem}"  # Use base_label_source

        ax.set_title(titleName, fontsize=18)

        # Create a unique filename for the plot, saved to plots_output_base_path
        # Combine base EOS name, the y-axis quantity, and make it filesystem-safe
        safe_ylabel = (
            ylabels[i]
            .replace(" ", "_")
            .replace("$", "")
            .replace("\\", "")
            .replace("{", "")
            .replace("}", "")
            .replace("^", "")
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        plot_file_name = f"{base_label_source.stem}_plot_{safe_ylabel}.png"  # Use base_label_source.stem

        # Save the plot to the correct directory
        plt.savefig(plots_output_base_path / plot_file_name)
        plt.close()  # Close the figure to free memory


def plot_S(S, plots_output_base_path):  # Added plots_output_base_path
    """
    Plots the singular values from DMD.

    Args:
        S (np.ndarray): Array of singular values.
        plots_output_base_path (Path or str): The base directory where plot files should be saved.
    """
    plots_output_base_path = Path(plots_output_base_path)
    plots_output_base_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Plot the S values:
    plt.plot(S / S[0], marker=".", markersize=10)
    plt.yscale("log")
    plt.xlabel("Index")
    plt.ylabel("S values")
    plt.savefig(plots_output_base_path / "Svalues.pdf")  # Updated save path
    plt.close()  # Use close instead of cla for cleaner resource management


def plot_parametric_old(
    Xdmd, X, name, tidal=False, xlim=None, ylim=None, plots_output_base_path=None
):  # Added plots_output_base_path
    """
    Makes plots for parametric DMD (old version).

    Args:
        Xdmd (np.ndarray): Reconstructed data.
        X (np.ndarray): Original data.
        name (str): Base name for the plot file.
        tidal (bool, optional): If True, indicates tidal data. Defaults to False.
        xlim (list, optional): X-axis limits for plots. Defaults to None.
        ylim (list[list], optional): Y-axis limits for plots. Defaults to None. Each inner list is [ymin, ymax].
        plots_output_base_path (Path or str, optional): The base directory where plot files should be saved.
                                                        Defaults to None (will use current directory if not provided).
    """
    plots_output_base_path = (
        Path(plots_output_base_path) if plots_output_base_path else Path.cwd()
    )
    plots_output_base_path.mkdir(parents=True, exist_ok=True)

    names = name.split("_")
    print("Names", names)
    print("Shape", Xdmd.shape, X.shape)
    if tidal:
        n = 2
    else:
        n = 1
    for i in range(n):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        color = colors[i]
        ax.plot(
            np.exp(Xdmd[0].real),
            np.exp(Xdmd[i + 2].real),
            color=color,
            label="DMD",
        )
        plt.plot(X[0], X[i + 2], ".", color=color, label="-data")

        if ylim is not None and i < len(ylim):
            ax.set_ylim(ylim[i])
        if xlim is not None and i < len(xlim):
            ax.set_xlim(xlim[i])

        plt.suptitle([f"p_{j} = {names[1+j]}" for j in range(len(names[1:]))])
        ax.set_xlabel(r"Radius (km)", fontsize=22)
        ax.set_ylabel(
            r"Mass $(M_{\odot})$", fontsize=22
        )  # This might need to be dynamic
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

        # Updated save paths
        if i == 0:
            plt.savefig(plots_output_base_path / f"MRpredict_{'_'.join(names)}.png")
        else:
            plt.savefig(plots_output_base_path / f"Tidalpredict_{'_'.join(names)}.png")
        plt.close()  # Use close instead of cla for cleaner resource management


def plot_parametric(
    Xdmd, X_exact, name, tidal, plots_output_base_path
):  # Added plots_output_base_path
    """
    Plots the reconstructed DMD data against the exact data for parametric runs.

    Args:
        Xdmd (np.ndarray): Reconstructed data (variables x snapshots).
        X_exact (np.ndarray): Exact data (variables x snapshots).
        name (str): Base name for the plot file.
        tidal (bool): Boolean, indicates if tidal data is being plotted (affects labels).
        plots_output_base_path (Path or str): The base directory where plot files should be saved.
    """
    plots_output_base_path = Path(plots_output_base_path)
    plots_output_base_path.mkdir(parents=True, exist_ok=True)

    num_snapshots = Xdmd.shape[1]
    time_points = np.arange(num_snapshots) * 1.0  # Assuming dt = 1.0

    fig, axs = plt.subplots(Xdmd.shape[0], 1, figsize=(10, 8), sharex=True)
    if Xdmd.shape[0] == 1:
        axs = [axs]

    if tidal:
        var_labels = ["R_b", "M_b", "k2_tidal"]  # Customize for your specific variables
    else:
        var_labels = [
            f"Variable {k+1}" for k in range(Xdmd.shape[0])  # Generic labels
        ]  # Default for non-tidal, adjust if needed

    for i in range(Xdmd.shape[0]):
        axs[i].plot(time_points, X_exact[i].real, label="Exact", color=colors[i])

        # Assuming Xdmd is log-scaled and needs np.exp
        axs[i].plot(
            time_points,
            np.exp(Xdmd[i].real),
            label="DMD Prediction",
            linestyle="--",
            color=colors[i + 3] if len(colors) > i + 3 else "k",
        )

        axs[i].set_ylabel(var_labels[i])
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Time Step")
    fig.suptitle(f'DMD Prediction vs Exact Data for {name.replace("Data_", "")}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot to the designated directory
    filename = plots_output_base_path / f"SLMpredict_{name}.png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    pass
