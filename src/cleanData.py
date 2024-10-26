"""
This code cleans up all the mess created by all the codes in this repo.
"""

import os
import shutil

# List of directories and file types to clean up
cleanup_targets = [
    "__pycache__",  # directory for compiled Python bytecode
    "*.pyc",  # compiled Python files
    "*.pyo",  # optimized compiled Python files
    ".DS_Store",  # macOS system file
    "build",  # build directory (if applicable)
    "dist",  # dist directory for Python packaging
    "*.egg-info",  # egg-info directories
    ".history",  # history files
]

# List of additional folders created by the code that you want to remove recursively
additional_folders_to_clean = [
    "EOS_files",  # log files directory
    "Results",  # output data directory
    "TOV_data",  # temporary data directory
    "testData",  # test data directory
    #"Plots",  # plots directory
    "trainData",  # training data directory
]


# Function to remove files or directories
def clean_directory(directory="."):
    for root, dirs, files in os.walk(
        directory, topdown=False
    ):  # Traverse the directory tree from bottom to top
        # Clean files matching patterns in cleanup_targets
        for target in cleanup_targets:
            for file in files:
                if file.endswith(target.replace("*", "")):
                    file_path = os.path.join(root, file)
                    print(f"Removing file: {file_path}")
                    os.remove(file_path)

        # Clean directories matching patterns in cleanup_targets or additional folders
        for dir in dirs:
            if dir in cleanup_targets or dir in additional_folders_to_clean:
                dir_path = os.path.join(root, dir)
                print(f"Removing directory: {dir_path}")
                shutil.rmtree(dir_path)


if __name__ == "__main__":
    clean_directory()  # Cleans the current directory by default
