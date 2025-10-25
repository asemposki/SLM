# src/SLM/config.py
import os
from pathlib import Path
import importlib.resources as pkg_resources

# Define the project root based on config.py's location.
# This assumes config.py is in src/SLM/, so it goes up three levels
# to reach the top-level directory (e.g., the one containing cno, dmdcodes, src, etc.)
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
    include_slm_paths: bool = True,  # New parameter to control SLM specific paths
) -> dict:
    """
    Returns a dictionary of resolved paths for data input/output and other project directories.
    Prioritizes explicit arguments for output_base_dir, then sensible defaults.
    Conditionally sets 'current' EOS/TOV/results paths based on the use_mseos flag.

    Args:
        output_base_dir (Path, optional): The base directory for all generated outputs
                                         and user-managed data. If None, defaults to
                                         the overall PROJECT_ROOT.
        use_mseos (bool): If True, 'current' paths will point to MSEOS-related directories.
                          If False, 'current' paths will point to QEOS-related directories.
                          Defaults to True.
        is_parametric_run (bool): If True, indicates a parametric run, affecting result/plot
                                  subdirectories. Defaults to True.
        include_slm_paths (bool): If True, SLM-specific results and plots directories
                                  will be included. If False, these paths are omitted.
                                  Defaults to True.

    Returns:
        dict: A dictionary containing all relevant path configurations.
    """
    # The overall project root, constant regardless of output_base_dir
    project_root = PROJECT_ROOT

    # The base directory for all generated outputs and user-managed data.
    # If output_base_dir is provided, use it. Otherwise, use the overall project_root.
    output_data_base = output_base_dir or project_root

    # The source directory, always relative to the overall project_root
    src_dir = project_root / DEFAULT_SRC_SUBDIR_NAME

    paths = {
        "project_root": project_root,
        "src_dir": src_dir,
        # Internal package resources (read-only, typically part of the installed package)
        # Assumes the package is named 'slmemulator' for pkg_resources.
        "package_eos_codes_dir": pkg_resources.files("slmemulator").joinpath(
            DEFAULT_EOS_CODES_SUBDIR_NAME
        ),
        "package_eos_data_dir": pkg_resources.files("slmemulator").joinpath(
            DEFAULT_EOS_DATA_SUBDIR_NAME
        ),
        # Directories for generated outputs and user-managed data, relative to output_data_base.
        # These are the directories your application will typically write to.
        "plots_dir": output_data_base / DEFAULT_PLOTS_SUBDIR_NAME,
        "results_dir": output_data_base / DEFAULT_RESULTS_SUBDIR_NAME,
        "docs_dir": output_data_base / DEFAULT_DOCS_SUBDIR_NAME,
        "tests_dir": output_data_base / DEFAULT_TESTS_SUBDIR_NAME,
        "tutorials_dir": output_data_base / DEFAULT_TUTORIALS_SUBDIR_NAME,
        # User-managed/project-level EOS and TOV data (can be inputs or outputs)
        "user_eos_data_dir": output_data_base / DEFAULT_EOS_DATA_SUBDIR_NAME,
        "user_tov_data_dir": output_data_base / DEFAULT_TOV_DATA_SUBDIR_NAME,
        "test_data_dir": output_data_base / DEFAULT_TEST_DATA_SUBDIR_NAME,
        "train_path": output_data_base / DEFAULT_TRAIN_DATA_SUBDIR_NAME,
        # Generated EOS files (e.g., from MSEOS.py/Quarkyonia.py)
        "generated_eos_files_dir": output_data_base / DEFAULT_EOS_FILES_SUBDIR_NAME,
        # Specific paths for QEOS and MSEOS generated files within generated_eos_files_dir
        "qeos_path_specific": (output_data_base / DEFAULT_EOS_FILES_SUBDIR_NAME)
        / "QEOS",
        "mseos_path_specific": (output_data_base / DEFAULT_EOS_FILES_SUBDIR_NAME)
        / "MSEOS",
        # Specific TOV data paths within user_tov_data_dir
        "qeos_tov_path_specific": (output_data_base / DEFAULT_TOV_DATA_SUBDIR_NAME)
        / "QEOS",
        "mseos_tov_path_specific": (output_data_base / DEFAULT_TOV_DATA_SUBDIR_NAME)
        / "MSEOS",
        # Removed DMD-related output directories from here. They will be user-defined.
    }

    # Conditionally add SLM-specific result/plot subdirectories
    if include_slm_paths:
        # General SLM specific result/plot subdirectories (non-parametric)
        paths["slm_res_mseos_specific"] = (
            output_data_base
            / DEFAULT_RESULTS_SUBDIR_NAME
            / "MSEOS"
            / DEFAULT_SLM_SUBDIR_NAME
        )
        paths["slm_res_qeos_specific"] = (
            output_data_base
            / DEFAULT_RESULTS_SUBDIR_NAME
            / "QEOS"
            / DEFAULT_SLM_SUBDIR_NAME
        )
        paths["slm_plots_mseos_specific"] = (
            output_data_base
            / DEFAULT_PLOTS_SUBDIR_NAME
            / "MSEOS"
            / DEFAULT_SLM_SUBDIR_NAME
        )
        paths["slm_plots_qeos_specific"] = (
            output_data_base
            / DEFAULT_PLOTS_SUBDIR_NAME
            / "QEOS"
            / DEFAULT_SLM_SUBDIR_NAME
        )

        # Parametric SLM specific result/plot subdirectories
        paths["slm_res_mseos_parametric_specific"] = (
            output_data_base
            / DEFAULT_RESULTS_SUBDIR_NAME
            / "MSEOS"
            / DEFAULT_PSLM_SUBDIR_NAME
        )
        paths["slm_res_qeos_parametric_specific"] = (
            output_data_base
            / DEFAULT_RESULTS_SUBDIR_NAME
            / "QEOS"
            / DEFAULT_PSLM_SUBDIR_NAME
        )
        paths["slm_plots_mseos_parametric_specific"] = (
            output_data_base
            / DEFAULT_PLOTS_SUBDIR_NAME
            / "MSEOS"
            / DEFAULT_PSLM_SUBDIR_NAME
        )
        paths["slm_plots_qeos_parametric_specific"] = (
            output_data_base
            / DEFAULT_PLOTS_SUBDIR_NAME
            / "QEOS"
            / DEFAULT_PSLM_SUBDIR_NAME
        )

    # Conditionally set the "current" paths based on the use_mseos flag AND `is_parametric_run`
    # These 'current' paths also depend on whether SLM paths are included.
    if use_mseos:
        paths["current_eos_input_dir"] = paths["mseos_path_specific"]
        paths["current_tov_data_dir"] = paths["mseos_tov_path_specific"]
        if include_slm_paths:  # Only set if SLM paths are desired
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
        if include_slm_paths:  # Only set if SLM paths are desired
            if is_parametric_run:
                paths["current_slm_results_dir"] = paths[
                    "slm_res_qeos_parametric_specific"
                ]
                paths["current_slm_plots_dir"] = paths[
                    "slm_plots_qeos_parametric_specific"
                ]
            else:  # Non-parametric SLM run for QEOS
                paths["current_slm_results_dir"] = paths["slm_res_qeos_specific"]
                paths["current_slm_plots_dir"] = paths["slm_plots_qeos_specific"]

    # Ensure all paths are Path objects, handling cases where a key might not be set
    # if include_slm_paths is False and a 'current_slm_...' path was attempted to be accessed.
    final_paths = {}
    for k, v in paths.items():
        if isinstance(v, Path):
            final_paths[k] = v

    return final_paths


def create_necessary_dirs(paths: dict, additional_dirs: list[Path] = None):
    """
    Creates directories for specified output and user-managed data paths if they do not already exist.
    This function focuses on creating directories that the application will write to.

    Args:
        paths (dict): A dictionary of Path objects, typically obtained from get_paths().
        additional_dirs (list[Path], optional): A list of additional Path objects
                                                to create (e.g., user-input directories).
                                                Defaults to None.
    """
    # Define a list of keys corresponding to directories that should be created.
    # These are typically output directories or user-managed data directories.
    # This list should be exhaustive for all directories you intend your application to write to.
    # Note: 'current_*' paths are included as they are the effective targets for writing.
    output_directory_keys = [
        "plots_dir",
        "results_dir",
        "docs_dir",  # If generated docs/reports go here
        "tests_dir",  # If test reports/output go here
        "tutorials_dir",  # If generated tutorial output goes here
        "user_eos_data_dir",
        "user_tov_data_dir",
        "test_data_dir",
        "train_path",
        "generated_eos_files_dir",
        "qeos_path_specific",
        "mseos_path_specific",
        "qeos_tov_path_specific",
        "mseos_tov_path_specific",
        # SLM specific paths (will only be present in 'paths' if include_slm_paths was True)
        "slm_res_mseos_specific",
        "slm_res_qeos_specific",
        "slm_plots_mseos_specific",
        "slm_plots_qeos_specific",
        "slm_res_mseos_parametric_specific",
        "slm_res_qeos_parametric_specific",
        "slm_plots_mseos_parametric_specific",
        "slm_plots_qeos_parametric_specific",
        # The 'current' paths are the effective targets, so ensure they are also created
        "current_eos_input_dir",
        "current_tov_data_dir",
        "current_slm_results_dir",
        "current_slm_plots_dir",
        # Removed "dmd_output_base_dir", "foo_files_dir", "dmd_input_dir" from here.
        # They will be passed dynamically via 'additional_dirs'.
    ]

    unique_dirs_to_create = set()
    for key in output_directory_keys:
        # Only add to the set if the key exists in 'paths' and its value is a Path object
        if key in paths and isinstance(paths[key], Path):
            unique_dirs_to_create.add(paths[key])

    # Add any explicitly provided additional directories
    if additional_dirs:
        for path_obj in additional_dirs:
            if isinstance(path_obj, Path):
                unique_dirs_to_create.add(path_obj)

    for path in unique_dirs_to_create:
        if not path.is_dir():
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {path}")
            except OSError as e:
                print(f"Error creating directory {path}: {e}")
