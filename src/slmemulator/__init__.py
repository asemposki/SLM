"""slmemulator: star log-extended emulation (SLM) based on dynamic mode decomposition."""

from .banach_grim import BanachGRIMInterpolator, rbf_kernel
from .emulator import (
    EOSDrawEmulator,
    TOVEmulator,
    load_eos_draws,
    load_tov_curves,
    mseos_generator,
    quarkyonia_generator,
)
from .SLM import SLM, augment_data_multiple_columns, solve_tov
from .TOV_class import TOV
from .config import get_paths, create_necessary_dirs
from .cleanData import clean_directory
from .plotData import plot_eigs, plot_slm, plot_slm_rad, plot_S, plot_parametric
from .pSLM import ParametricSLM, gaussian_kernel
from .recombination import gaussian_rbf, recombination_thinning
from .EOS_Codes import MSEOS, generate_quarkyonia_eos

__version__ = "0.1.0"

__all__ = [
    "BanachGRIMInterpolator",
    "rbf_kernel",
    "TOVEmulator",
    "EOSDrawEmulator",
    "load_eos_draws",
    "load_tov_curves",
    "quarkyonia_generator",
    "mseos_generator",
    "SLM",
    "augment_data_multiple_columns",
    "solve_tov",
    "TOV",
    "get_paths",
    "create_necessary_dirs",
    "clean_directory",
    "plot_eigs",
    "plot_slm",
    "plot_slm_rad",
    "plot_S",
    "plot_parametric",
    "ParametricSLM",
    "gaussian_kernel",
    "gaussian_rbf",
    "recombination_thinning",
    "MSEOS",
    "generate_quarkyonia_eos",
    "__version__",
]
