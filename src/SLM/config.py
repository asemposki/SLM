# src/SLM/config.py
import os
from pathlib import Path


# Function to get current working directory for defaults
def _get_cwd_safe() -> Path:
    """Returns the current working directory as a Path object."""
    return Path.cwd()


# Default paths for generated outputs (relative to current working directory of execution)
DEFAULT_OUTPUT_BASE_DIR = (
    _get_cwd_safe()
)  # Default output base is the current working directory

DEFAULT_EOS_DATA_SUBDIR_NAME = "EOS_Data"
DEFAULT_PLOTS_SUBDIR_NAME = "plots"
DEFAULT_RESULTS_SUBDIR_NAME = "Results"
DEFAULT_TOV_DATA_SUBDIR_NAME = "TOV_data"
DEFAULT_TEST_DATA_SUBDIR_NAME = "testData"

# Default path for external input data (can be None to force user config)
DEFAULT_EOS_FILES_SUBDIR_NAME = "EOS_files"


def get_paths(
    output_base_dir: Path = None,
    eos_files_dir: Path = None,
    # Add other top-level directories if they can be independently set
) -> dict:
    """
    Returns a dictionary of resolved paths for data input/output.
    Prioritizes explicit arguments, then environment variables, then sensible defaults.

    Args:
        output_base_dir (Path, optional): The base directory for all generated outputs.
                                         Defaults to current working directory.
        eos_files_dir (Path, optional): The directory containing external EOS input files.
                                        Defaults to a subdirectory within the current working directory.

    Returns:
        dict: A dictionary containing resolved Path objects for all relevant directories.
    """
    # 1. Prioritize direct function arguments
    output_base = output_base_dir if output_base_dir else DEFAULT_OUTPUT_BASE_DIR
    eos_files_base = (
        eos_files_dir
        if eos_files_dir
        else (DEFAULT_OUTPUT_BASE_DIR / DEFAULT_EOS_FILES_SUBDIR_NAME)
    )

    # 2. Check Environment Variables (higher priority than simple defaults)
    output_base = Path(os.getenv("SLM_OUTPUT_BASE_DIR", str(output_base))).resolve()
    eos_files_base = Path(os.getenv("SLM_EOS_FILES_DIR", str(eos_files_base))).resolve()

    # Construct final paths
    paths = {
        "output_base_dir": output_base,  # The overall base for generated outputs
        "eos_data_dir": output_base / DEFAULT_EOS_DATA_SUBDIR_NAME,
        "plots_dir": output_base / DEFAULT_PLOTS_SUBDIR_NAME,
        "results_dir": output_base / DEFAULT_RESULTS_SUBDIR_NAME,
        "tov_data_dir": output_base / DEFAULT_TOV_DATA_SUBDIR_NAME,
        "test_data_dir": output_base
        / DEFAULT_TEST_DATA_SUBDIR_NAME,  # For generated test data
        "eos_files_dir": eos_files_base,  # This is the input data directory
        # Specific sub-paths for EOS_files and TOV_data (relative to their respective base paths)
        "qeos_path": eos_files_base / "QEOS",
        "mseos_path": eos_files_base / "MSEOS",
        "qeos_tov_path": (output_base / DEFAULT_TOV_DATA_SUBDIR_NAME) / "QEOS",
        "mseos_tov_path": (output_base / DEFAULT_TOV_DATA_SUBDIR_NAME) / "MSEOS",
        "slm_res_mseos": (output_base / DEFAULT_RESULTS_SUBDIR_NAME) / "MSEOS",
        "slm_res_qeos": (output_base / DEFAULT_RESULTS_SUBDIR_NAME) / "QEOS",
    }

    # Ensure all paths are Path objects
    return {k: Path(v) for k, v in paths.items()}


# Example usage (for your main scripts/CLI):
# from slm_package.config import get_paths
#
# if __name__ == "__main__":
#     # User can specify via CLI, config file, or env vars
#     # For this example, we just use defaults:
#     current_paths = get_paths()
#     print(f"EOS Data will be written to: {current_paths['eos_data_dir']}")
#     current_paths['eos_data_dir'].mkdir(parents=True, exist_ok=True)
#     # ... your code uses these paths to read/write ...
