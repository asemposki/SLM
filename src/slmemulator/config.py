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
DEFAULT_TEST_DATA_SUBDIR_NAME = "testData" # This is the base for test/QEOS/pSLM
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
# Default subdirectory name for non-Parametric SLM results/plots
DEFAULT_SLM_SUBDIR_NAME = "SLM"


def get_paths(
    output_base_dir: Path = None,
    eos_name: str = "MSEOS",
    is_parametric_run: bool = True,
    include_slm_paths: bool = True,
) -> dict:
    """
    Returns a dictionary of resolved paths for data input/output and other project directories.
    All 'current' paths are dynamically set based on the provided 'eos_name'.

    Args:
        output_base_dir (Path, optional): The base directory for all generated outputs.
        eos_name (str): The name of the Equation of State (e.g., "MSEOS", "QEOS", "APR").
                        This name is used to create specific subdirectories. Defaults to "MSEOS".
        is_parametric_run (bool): If True, affects result/plot/test subdirectories (pSLM vs SLM).
        include_slm_paths (bool): If True, SLM-specific results, plots, and test directories are included.

    Returns:
        dict: A dictionary containing all relevant path configurations.
    """
    project_root = PROJECT_ROOT
    output_data_base = output_base_dir or project_root
    src_dir = project_root / DEFAULT_SRC_SUBDIR_NAME
    
    # Define extensions to strip
    EXTENSIONS_TO_STRIP = [".txt", ".dat", ".table"]
    
    # Strip extensions and convert to uppercase for directory naming
    clean_eos_name = eos_name
    for ext in EXTENSIONS_TO_STRIP:
        if clean_eos_name.lower().endswith(ext):
            clean_eos_name = clean_eos_name[:-len(ext)]
            break # Stop after stripping the first matching extension
            
    # --- Dynamic EOS-specific folder name, ensuring it's uppercase for directories ---
    EOS_FOLDER_NAME = clean_eos_name.upper()
    
    # --- Define SLM subdirectory based on run type (SLM or pSLM) ---
    SLM_SUBDIR = DEFAULT_PSLM_SUBDIR_NAME if is_parametric_run else DEFAULT_SLM_SUBDIR_NAME

    paths = {
        "project_root": project_root,
        "src_dir": src_dir,
        # Internal package resources (read-only)
        "package_eos_codes_dir": pkg_resources.files("slmemulator").joinpath(
            DEFAULT_EOS_CODES_SUBDIR_NAME
        ),
        "package_eos_data_dir": pkg_resources.files("slmemulator").joinpath(
            DEFAULT_EOS_DATA_SUBDIR_NAME
        ),
        # General output directories
        "plots_dir": output_data_base / DEFAULT_PLOTS_SUBDIR_NAME,
        "results_dir": output_data_base / DEFAULT_RESULTS_SUBDIR_NAME,
        "docs_dir": output_data_base / DEFAULT_DOCS_SUBDIR_NAME,
        "tests_dir": output_data_base / DEFAULT_TESTS_SUBDIR_NAME,
        "tutorials_dir": output_data_base / DEFAULT_TUTORIALS_SUBDIR_NAME,
        
        # User-managed/project-level EOS and TOV data
        "user_eos_data_dir": output_data_base / DEFAULT_EOS_DATA_SUBDIR_NAME,
        "test_data_dir": output_data_base / DEFAULT_TEST_DATA_SUBDIR_NAME,
        "train_path": output_data_base / DEFAULT_TRAIN_DATA_SUBDIR_NAME,
        "generated_eos_files_dir": output_data_base / DEFAULT_EOS_FILES_SUBDIR_NAME,
        
        # Current Dynamic EOS/TOV/Generated paths
        "current_eos_input_dir": (
            output_data_base / DEFAULT_EOS_FILES_SUBDIR_NAME
        ) / EOS_FOLDER_NAME,
        "current_tov_data_dir": (
            output_data_base / DEFAULT_TOV_DATA_SUBDIR_NAME
        ) / EOS_FOLDER_NAME,
        
        # Kept old names for minimal compatibility changes
        "qeos_path_specific": (output_data_base / DEFAULT_EOS_FILES_SUBDIR_NAME) / "QEOS",
        "mseos_path_specific": (output_data_base / DEFAULT_EOS_FILES_SUBDIR_NAME) / "MSEOS",
        "qeos_tov_path_specific": (output_data_base / DEFAULT_TOV_DATA_SUBDIR_NAME) / "QEOS",
        "mseos_tov_path_specific": (output_data_base / DEFAULT_TOV_DATA_SUBDIR_NAME) / "MSEOS",
        "user_tov_data_dir": output_data_base / DEFAULT_TOV_DATA_SUBDIR_NAME,
    }

    # Conditionally add SLM-specific result/plot/test subdirectories
    if include_slm_paths:
        
        # Define the dynamic current SLM results and plots paths
        paths["current_slm_results_dir"] = (
            output_data_base
            / DEFAULT_RESULTS_SUBDIR_NAME
            / EOS_FOLDER_NAME
            / SLM_SUBDIR
        )
        paths["current_slm_plots_dir"] = (
            output_data_base
            / DEFAULT_PLOTS_SUBDIR_NAME
            / EOS_FOLDER_NAME
            / SLM_SUBDIR
        )

        # --- NEW PATH: Dynamic SLM Test Data Path ---
        if is_parametric_run:
            # Only create the /pSLM folder for tests if it's a parametric run
            paths["current_slm_tests_dir"] = (
                output_data_base
                / DEFAULT_TEST_DATA_SUBDIR_NAME
                / EOS_FOLDER_NAME
                / DEFAULT_PSLM_SUBDIR_NAME  # Always use 'pSLM' for parametric tests
            )
        else:
            # If not parametric, use the base testData/EOS_NAME folder
            paths["current_slm_tests_dir"] = (
                output_data_base
                / DEFAULT_TEST_DATA_SUBDIR_NAME
                / EOS_FOLDER_NAME
            )
    
    # Ensure all paths are Path objects and filter out any None/non-Path values
    final_paths = {}
    for k, v in paths.items():
        if isinstance(v, Path):
            final_paths[k] = v
        elif include_slm_paths and k.startswith("current_slm_") and k in paths:
             final_paths[k] = paths[k]

    return final_paths


def create_necessary_dirs(paths: dict, additional_dirs: list[Path] = None):
    """
    Creates directories for specified output and user-managed data paths if they do not already exist.
    """
    
    # Define a list of keys corresponding to directories that should be created.
    output_directory_keys = [
        "plots_dir",
        "results_dir",
        "docs_dir",
        "tests_dir",
        "tutorials_dir",
        "user_eos_data_dir",
        "user_tov_data_dir",
        "test_data_dir",
        "train_path",
        "generated_eos_files_dir",
        
        # The 'current' paths are the effective targets, so ensure they are created
        "current_eos_input_dir",
        "current_tov_data_dir",
        "current_slm_results_dir",
        "current_slm_plots_dir",
        "current_slm_tests_dir", # <-- ADDED NEW PATH
        
        # Kept for backward compatibility
        "qeos_path_specific",
        "mseos_path_specific",
        "qeos_tov_path_specific",
        "mseos_tov_path_specific",
    ]

    unique_dirs_to_create = set()
    for key in output_directory_keys:
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