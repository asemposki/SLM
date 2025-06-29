###########################################
# Clean up the directories
# Author: Sudhanva Lalit
# Last edited: 24 November 2024
###########################################
import os
import shutil
import sys
from pathlib import Path

# Import default subdirectory names from config for consistent cleanup targets
from slmemulator.config import (
    DEFAULT_EOS_FILES_SUBDIR_NAME,
    DEFAULT_RESULTS_SUBDIR_NAME,
    DEFAULT_TOV_DATA_SUBDIR_NAME,
    DEFAULT_TEST_DATA_SUBDIR_NAME,
    DEFAULT_TRAIN_DATA_SUBDIR_NAME,
    DEFAULT_PLOTS_SUBDIR_NAME,
)

# List of common cleanup targets (directories and file extensions)
cleanup_targets = [
    "__pycache__",  # directory for compiled Python bytecode
    ".pyc",  # compiled Python files
    ".pyo",  # optimized compiled Python files
    ".DS_Store",  # macOS system file
    "build",  # build directory (if applicable)
    "dist",  # dist directory for Python packaging
    ".egg-info",  # egg-info directories
    ".history",  # history files
]


def clean_directory(directory: str = None):
    """
    Cleans up specified files and directories within a given directory.

    Parameters:
    directory (str): The path to the directory to clean. Defaults to the
                     current working directory if None.
    """
    if directory is None:
        directory = os.getcwd()

    # Ensure the target directory exists and is a directory
    target_dir_path = Path(directory).resolve()
    if not target_dir_path.is_dir():
        print(f"Error: Directory '{directory}' not found or is not a directory.")
        return

    print(f"Cleaning directory: {target_dir_path}")

    # List of additional folders created by the code that you want to remove recursively.
    # These are now relative to the 'directory' argument.
    additional_folders_to_clean_names = [
        DEFAULT_EOS_FILES_SUBDIR_NAME,
        DEFAULT_RESULTS_SUBDIR_NAME,
        DEFAULT_TOV_DATA_SUBDIR_NAME,
        DEFAULT_TEST_DATA_SUBDIR_NAME,
        DEFAULT_TRAIN_DATA_SUBDIR_NAME,
        DEFAULT_PLOTS_SUBDIR_NAME,
    ]

    # Convert these names to full paths within the target directory
    additional_folders_to_clean_paths = [
        target_dir_path / name for name in additional_folders_to_clean_names
    ]

    for root, dirs, files in os.walk(
        target_dir_path, topdown=False
    ):  # Traverse from bottom to top
        # Clean files matching patterns in cleanup_targets
        for file in files:
            for target_ext in [
                t for t in cleanup_targets if t.startswith(".")
            ]:  # Only check extensions
                if file.endswith(target_ext):
                    file_path = os.path.join(root, file)
                    print(f"Removing file: {file_path}")
                    os.remove(file_path)

        # Clean directories matching patterns in cleanup_targets (by name)
        # or directories matching full paths in additional_folders_to_clean_paths
        for dir_name in dirs:
            dir_full_path = Path(root) / dir_name  # Use Path for comparison
            if (
                dir_name in cleanup_targets
                or dir_full_path in additional_folders_to_clean_paths
            ):
                print(f"Removing directory: {dir_full_path}")
                shutil.rmtree(dir_full_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean up specified files and directories."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=None,
        help="The directory to clean. Defaults to the current working directory.",
    )
    args = parser.parse_args()

    # Get user confirmation before proceeding
    confirm = input("Clean up the directories? (True/False)[False]: ").strip().lower()
    if confirm == "true":
        clean_directory(args.directory)
    else:
        print("Cleanup aborted.")
