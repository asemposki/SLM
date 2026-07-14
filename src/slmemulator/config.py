"""Project path configuration for slmemulator."""

from importlib import resources
from pathlib import Path

__all__ = [
    "get_paths",
    "create_necessary_dirs",
]

# Repository root when running from a source checkout (three levels above
# this file: src/slmemulator/config.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default names for the output subdirectories
DEFAULT_SRC_SUBDIR_NAME = "src"
DEFAULT_EOS_DATA_SUBDIR_NAME = "EOS_Data"
DEFAULT_PLOTS_SUBDIR_NAME = "Plots"
DEFAULT_RESULTS_SUBDIR_NAME = "Results"
DEFAULT_TOV_DATA_SUBDIR_NAME = "TOV_data"
DEFAULT_TEST_DATA_SUBDIR_NAME = "testData"  # base for test/QEOS/pSLM
DEFAULT_TRAIN_DATA_SUBDIR_NAME = "trainData"
DEFAULT_VAL_DATA_SUBDIR_NAME = "valData"
DEFAULT_DOCS_SUBDIR_NAME = "docs"
DEFAULT_TESTS_SUBDIR_NAME = "tests"
DEFAULT_TUTORIALS_SUBDIR_NAME = "Tutorials"
DEFAULT_EOS_CODES_SUBDIR_NAME = "EOS_Codes"  # EOS generation scripts
DEFAULT_EOS_FILES_SUBDIR_NAME = "EOS_files"  # generated/processed EOS files

# Subdirectory names for parametric vs. non-parametric SLM outputs
DEFAULT_PSLM_SUBDIR_NAME = "pSLM"
DEFAULT_SLM_SUBDIR_NAME = "SLM"

# File extensions stripped from an EOS name when building directory names
_EXTENSIONS_TO_STRIP = (".txt", ".dat", ".table")


def get_paths(
    output_base_dir: Path | None = None,
    eos_name: str = "MSEOS",
    is_parametric_run: bool = True,
    include_slm_paths: bool = True,
) -> dict[str, Path]:
    """
    Generates and returns a dictionary of resolved project paths, dynamically
    structuring subdirectories based on the Equation of State (EOS) name and
    run configuration.

    The function provides paths for input data, model binaries, general output,
    and specific subdirectories for results, plots, and test data related to
    SLM (Star Log-extended eMulator) or pSLM (parametric SLM) runs.

    Parameters:
        output_base_dir (pathlib.Path, optional):
            The root directory where all generated project outputs (results,
            plots, test data) will be stored. If ``None``, the repository root
            is used. Defaults to ``None``.
        eos_name (str, optional):
            The name of the Equation of State (e.g., "MSEOS", "QEOS", "APR").
            This name dictates the specific subdirectory created for the
            current run within the results, plots, and test directories.
            Defaults to "MSEOS".
        is_parametric_run (bool, optional):
            Flag indicating if the current modeling run is using the parametric
            SLM (pSLM) approach. If ``True``, the output paths will include
            'pSLM'; otherwise 'SLM'. Defaults to ``True``.
        include_slm_paths (bool, optional):
            If ``True``, the dictionary includes the run-specific directories
            (``current_slm_results_dir``, ``current_slm_plots_dir``,
            ``current_slm_tests_dir``). Defaults to ``True``.

    Returns:
        dict[str, pathlib.Path]: A dictionary containing all relevant path
        configurations.
    """
    output_data_base = output_base_dir or PROJECT_ROOT
    src_dir = PROJECT_ROOT / DEFAULT_SRC_SUBDIR_NAME

    # Strip a known extension and upper-case the EOS name for directory naming
    clean_eos_name = eos_name
    for ext in _EXTENSIONS_TO_STRIP:
        if clean_eos_name.lower().endswith(ext):
            clean_eos_name = clean_eos_name[: -len(ext)]
            break
    eos_folder_name = clean_eos_name.upper()

    slm_subdir = (
        DEFAULT_PSLM_SUBDIR_NAME if is_parametric_run else DEFAULT_SLM_SUBDIR_NAME
    )

    # Concrete Path is fine here: the package is installed on the filesystem
    # (packaged EOS data could not be read through np.loadtxt otherwise).
    package_root = Path(str(resources.files("slmemulator")))

    paths: dict[str, Path] = {
        "project_root": PROJECT_ROOT,
        "src_dir": src_dir,
        # Internal package resources (read-only)
        "package_eos_codes_dir": package_root.joinpath(DEFAULT_EOS_CODES_SUBDIR_NAME),
        "package_eos_data_dir": package_root.joinpath(DEFAULT_EOS_DATA_SUBDIR_NAME),
        # General output directories
        "plots_dir": output_data_base / DEFAULT_PLOTS_SUBDIR_NAME,
        "results_dir": output_data_base / DEFAULT_RESULTS_SUBDIR_NAME,
        "docs_dir": output_data_base / DEFAULT_DOCS_SUBDIR_NAME,
        "tests_dir": output_data_base / DEFAULT_TESTS_SUBDIR_NAME,
        "tutorials_dir": output_data_base / DEFAULT_TUTORIALS_SUBDIR_NAME,
        # User-managed/project-level EOS and TOV data
        "user_eos_data_dir": output_data_base / DEFAULT_EOS_DATA_SUBDIR_NAME,
        "test_data_dir": output_data_base / DEFAULT_TEST_DATA_SUBDIR_NAME,
        "train_data_dir": output_data_base / DEFAULT_TRAIN_DATA_SUBDIR_NAME,
        "val_data_dir": output_data_base / DEFAULT_VAL_DATA_SUBDIR_NAME,
        "generated_eos_files_dir": output_data_base / DEFAULT_EOS_FILES_SUBDIR_NAME,
        # Current dynamic EOS/TOV/generated paths
        "current_eos_input_dir": output_data_base
        / DEFAULT_EOS_FILES_SUBDIR_NAME
        / eos_folder_name,
        "current_tov_data_dir": output_data_base
        / DEFAULT_TOV_DATA_SUBDIR_NAME
        / eos_folder_name,
        "current_train_data_dir": output_data_base
        / DEFAULT_TRAIN_DATA_SUBDIR_NAME
        / eos_folder_name,
        "current_val_data_dir": output_data_base
        / DEFAULT_VAL_DATA_SUBDIR_NAME
        / eos_folder_name,
        "current_test_data_dir": output_data_base
        / DEFAULT_TEST_DATA_SUBDIR_NAME
        / eos_folder_name,
        # Kept old names for minimal compatibility changes
        "qeos_path_specific": output_data_base / DEFAULT_EOS_FILES_SUBDIR_NAME / "QEOS",
        "mseos_path_specific": output_data_base
        / DEFAULT_EOS_FILES_SUBDIR_NAME
        / "MSEOS",
        "qeos_tov_path_specific": output_data_base
        / DEFAULT_TOV_DATA_SUBDIR_NAME
        / "QEOS",
        "mseos_tov_path_specific": output_data_base
        / DEFAULT_TOV_DATA_SUBDIR_NAME
        / "MSEOS",
        "user_tov_data_dir": output_data_base / DEFAULT_TOV_DATA_SUBDIR_NAME,
    }

    if include_slm_paths:
        paths["current_slm_results_dir"] = (
            output_data_base / DEFAULT_RESULTS_SUBDIR_NAME / eos_folder_name / slm_subdir
        )
        paths["current_slm_plots_dir"] = (
            output_data_base / DEFAULT_PLOTS_SUBDIR_NAME / eos_folder_name / slm_subdir
        )
        if is_parametric_run:
            # Parametric tests always live under testData/<EOS>/pSLM
            paths["current_slm_tests_dir"] = (
                output_data_base
                / DEFAULT_TEST_DATA_SUBDIR_NAME
                / eos_folder_name
                / DEFAULT_PSLM_SUBDIR_NAME
            )
        else:
            paths["current_slm_tests_dir"] = (
                output_data_base / DEFAULT_TEST_DATA_SUBDIR_NAME / eos_folder_name
            )

    return paths


def create_necessary_dirs(
    paths: dict[str, Path],
    additional_dirs: list[Path] | None = None,
) -> None:
    """
    Creates necessary directories specified in a dictionary and an optional list.

    Iterates through the known output-directory keys of ``paths`` (plus any
    ``additional_dirs``) and creates each directory if it does not already
    exist (``mkdir(parents=True, exist_ok=True)``).

    Parameters:
        paths (dict[str, pathlib.Path]): A dictionary as returned by
            :func:`get_paths`, mapping string identifiers to directories.
        additional_dirs (list[pathlib.Path], optional): Additional directories
            to create (e.g., user-managed data or model directories).
            Defaults to None.

    Returns:
        None: The function modifies the filesystem but does not return a value.
    """
    output_directory_keys = [
        "plots_dir",
        "results_dir",
        "docs_dir",
        "tests_dir",
        "tutorials_dir",
        "user_eos_data_dir",
        "user_tov_data_dir",
        "test_data_dir",
        "train_data_dir",
        "generated_eos_files_dir",
        # The 'current' paths are the effective targets
        "current_eos_input_dir",
        "current_tov_data_dir",
        "current_slm_results_dir",
        "current_slm_plots_dir",
        "current_slm_tests_dir",
        # Kept for backward compatibility
        "qeos_path_specific",
        "mseos_path_specific",
        "qeos_tov_path_specific",
        "mseos_tov_path_specific",
    ]

    dirs_to_create = {
        paths[key]
        for key in output_directory_keys
        if key in paths and isinstance(paths[key], Path)
    }
    if additional_dirs:
        dirs_to_create.update(p for p in additional_dirs if isinstance(p, Path))

    for path in dirs_to_create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {path}: {e}")
