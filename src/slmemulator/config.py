# src/SLM/config.py
import os
from pathlib import Path
import importlib.resources as pkg_resources  # New import

# Define the project root based on config.py's location
PROJECT_ROOT = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Default path names for subdirectories
DEFAULT_SRC_SUBDIR_NAME = "src"
DEFAULT_EOS_DATA_SUBDIR_NAME = "EOS_Data"
DEFAULT_PLOTS_SUBDIR_NAME = "Plots"
DEFAULT_RESULTS_SUBDIR_NAME = "Results"
DEFAULT_TOV_DATA_SUBDIR_NAME = "TOV_data"
DEFAULT_TEST_DATA_SUBDIR_NAME = "testData"
DEFAULT_TRAIN_DATA_SUBDIR_NAME = "trainData"
DEFAULT_DOCS_SUBDIR_NAME = "docs"
DEFAULT_TESTS_SUBDIR_NAME = "tests"
DEFAULT_TUTORIALS_SUBDIR_NAME = "Tutorials"
DEFAULT_EOS_CODES_SUBDIR_NAME = (
    "EOS_Codes"  # The directory containing EOS generation scripts
)
DEFAULT_EOS_FILES_SUBDIR_NAME = "EOS_files"  # For generated/processed EOS files

# Default subdirectory name for Parametric SLM results/plots
DEFAULT_PSLM_SUBDIR_NAME = "pSLM"
# NEW: Default subdirectory name for non-Parametric SLM results/plots
DEFAULT_SLM_SUBDIR_NAME = "SLM"


def get_paths(
    output_base_dir: Path = None,
    use_mseos: bool = True,
    is_parametric_run: bool = True,
) -> dict:
    """
    Returns a dictionary of resolved paths for data input/output and other project directories.
    Prioritizes explicit arguments for output_base_dir, then sensible defaults.
    Conditionally sets 'current' EOS/TOV/results paths based on the use_mseos flag.
    Adjusts base for general data directories (EOS_Data, TOV_data, testData, trainData)
    to output_base_dir if provided (for local working directories),
    otherwise defaults them to be under src_dir (for package internal use).

    Args:
        output_base_dir (Path, optional): The base directory for all generated outputs.
                                         Defaults to the project root.
        use_mseos (bool): If True, 'current' paths will point to MSEOS-related directories.
                          If False, 'current' paths will point to QEOS-related directories.
                          Defaults to True.
        is_parametric_run (bool): If True, indicates a parametric run.
                                  This parameter is available for future path logic,
                                  but currently the primary factor for
                                  data directory location is `output_base_dir`.
                                  Defaults to True.

    Returns:
        dict: A dictionary containing all relevant path configurations.
    """
    # Use explicit output_base_dir or default to the calculated PROJECT_ROOT
    output_base = output_base_dir or PROJECT_ROOT

    # src_dir will always be under the primary base (output_base or PROJECT_ROOT)
    src_dir = output_base / DEFAULT_SRC_SUBDIR_NAME

    # Determine the base for general data directories (EOS_Data, TOV_data, testData, trainData)
    # If output_base_dir is explicitly provided, these data directories should be
    # relative to it. Otherwise, they remain within the src_dir of the package.
    if output_base_dir is not None:
        data_root_dir = output_base
    else:
        data_root_dir = src_dir

    paths = {
        "project_root": output_base,
        "src_dir": src_dir,
        # EOS_Codes_dir should point to the location within the *installed package*
        # This ensures that whether installed or in editable mode, it finds the right place.
        "eos_codes_dir": pkg_resources.files("slmemulator").joinpath(
            DEFAULT_EOS_CODES_SUBDIR_NAME
        ),
        "plots_dir": data_root_dir / DEFAULT_PLOTS_SUBDIR_NAME,
        "results_dir": data_root_dir / DEFAULT_RESULTS_SUBDIR_NAME,
        "docs_dir": output_base / DEFAULT_DOCS_SUBDIR_NAME,
        "tests_dir": output_base / DEFAULT_TESTS_SUBDIR_NAME,
        "tutorials_dir": output_base / DEFAULT_TUTORIALS_SUBDIR_NAME,
        # Data directories whose location depends on `output_base_dir`
        "eos_data_dir": data_root_dir
        / DEFAULT_EOS_DATA_SUBDIR_NAME,  # For user-provided/existing EOS data
        "tov_data_dir": data_root_dir / DEFAULT_TOV_DATA_SUBDIR_NAME,
        "test_data_dir": data_root_dir / DEFAULT_TEST_DATA_SUBDIR_NAME,
        "train_path": data_root_dir / DEFAULT_TRAIN_DATA_SUBDIR_NAME,
        # EOS_files_dir contains generated EOS files (MSEOS/QEOS subdirs)
        # This is where output of MSEOS.py/Quarkyonia.py goes, typically under src/
        "eos_files_dir": data_root_dir / DEFAULT_EOS_FILES_SUBDIR_NAME,
        "eos_data_dir": pkg_resources.files("slmemulator").joinpath(
            DEFAULT_EOS_DATA_SUBDIR_NAME
        ),
        # Specific paths for QEOS and MSEOS (always defined relative to their respective bases)
        "qeos_path_specific": (data_root_dir / DEFAULT_EOS_FILES_SUBDIR_NAME) / "QEOS",
        "mseos_path_specific": (data_root_dir / DEFAULT_EOS_FILES_SUBDIR_NAME)
        / "MSEOS",
        # Correcting specific TOV data paths to use data_root_dir (for user data)
        "qeos_tov_path_specific": (data_root_dir / DEFAULT_TOV_DATA_SUBDIR_NAME)
        / "QEOS",
        "mseos_tov_path_specific": (data_root_dir / DEFAULT_TOV_DATA_SUBDIR_NAME)
        / "MSEOS",
        # General SLM specific result/plot subdirectories (non-parametric)
        "slm_res_mseos_specific": (
            data_root_dir
            / DEFAULT_RESULTS_SUBDIR_NAME
            / "MSEOS"
            / DEFAULT_SLM_SUBDIR_NAME
        ),
        "slm_res_qeos_specific": (
            data_root_dir
            / DEFAULT_RESULTS_SUBDIR_NAME
            / "QEOS"
            / DEFAULT_SLM_SUBDIR_NAME
        ),
        "slm_plots_mseos_specific": (
            data_root_dir
            / DEFAULT_PLOTS_SUBDIR_NAME
            / "MSEOS"
            / DEFAULT_SLM_SUBDIR_NAME
        ),
        "slm_plots_qeos_specific": (
            data_root_dir / DEFAULT_PLOTS_SUBDIR_NAME / "QEOS" / DEFAULT_SLM_SUBDIR_NAME
        ),
        # Parametric SLM specific result/plot subdirectories
        "slm_res_mseos_parametric_specific": (
            data_root_dir
            / DEFAULT_RESULTS_SUBDIR_NAME
            / "MSEOS"
            / DEFAULT_PSLM_SUBDIR_NAME
        ),
        "slm_res_qeos_parametric_specific": (
            data_root_dir
            / DEFAULT_RESULTS_SUBDIR_NAME
            / "QEOS"
            / DEFAULT_PSLM_SUBDIR_NAME
        ),
        "slm_plots_mseos_parametric_specific": (
            data_root_dir
            / DEFAULT_PLOTS_SUBDIR_NAME
            / "MSEOS"
            / DEFAULT_PSLM_SUBDIR_NAME
        ),
        "slm_plots_qeos_parametric_specific": (
            data_root_dir
            / DEFAULT_PLOTS_SUBDIR_NAME
            / "QEOS"
            / DEFAULT_PSLM_SUBDIR_NAME
        ),
    }

    # Conditionally set the "current" paths based on the use_mseos flag AND `is_parametric_run`
    if use_mseos:
        paths["current_eos_input_dir"] = paths["mseos_path_specific"]
        paths["current_tov_data_dir"] = paths["mseos_tov_path_specific"]
        if is_parametric_run:
            paths["current_slm_results_dir"] = paths[
                "slm_res_mseos_parametric_specific"
            ]
            paths["current_slm_plots_dir"] = paths[
                "slm_plots_mseos_parametric_specific"
            ]
        else:  # Non-parametric SLM run for MSEOS
            paths["current_slm_results_dir"] = paths["slm_res_mseos_specific"]
            paths["current_slm_plots_dir"] = paths["slm_plots_mseos_specific"]
    else:  # QEOS
        paths["current_eos_input_dir"] = paths["qeos_path_specific"]
        paths["current_tov_data_dir"] = paths["qeos_tov_path_specific"]
        if is_parametric_run:
            paths["current_slm_results_dir"] = paths["slm_res_qeos_parametric_specific"]
            paths["current_slm_plots_dir"] = paths["slm_plots_qeos_parametric_specific"]
        else:  # Non-parametric SLM run for QEOS
            paths["current_slm_results_dir"] = paths["slm_res_qeos_specific"]
            paths["current_slm_plots_dir"] = paths["slm_plots_qeos_specific"]

    # Ensure all paths are Path objects
    return {k: Path(v) for k, v in paths.items()}
