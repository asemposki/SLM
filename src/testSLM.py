###########################################
# Test script for parametric or general DMD
# Author: Sudhanva Lalit
# Last edited: 24 November 2024
###########################################

import os
import subprocess
import sys
import numpy as np
from cleanData import clean_directory

# Ensure that the parent directory (project root) is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from src import (
    SRC_DIR,
    EOS_CODES_DIR,
    EOS_DATA_DIR,
    EOS_FILES_DIR,
    RESULTS_PATH,
    TOV_PATH,
    PLOTS_PATH,
    MSEOS_PATH,
    QEOS_PATH,
    SLM_RES_MSEOS,
    SLM_RES_QEOS,
)

# set parameters for lambda and kappa
lamVal = np.linspace(300, 500, 20)
kappaVal = np.linspace(0.1, 0.3, 10)
Ls = np.linspace(0.0, 3e-3, 4)
Lv = np.linspace(0.0, 3e-2, 4)
zetaVal = np.linspace(1e-4, 2e-4, 2)
xiVal = np.linspace(0.0, 1.0, 2)  # Original value in first file was 1.0, not 1e-4

print(f"Total number of runs: {len(lamVal) * len(kappaVal)}")
# lamVal = [363.16]
# kappaVal = [0.23]

def _run_system_command(command, cwd=None):
    """
    Helper function to run an os.system command with optional directory change.
    Prints the command being executed.
    """
    original_cwd = os.getcwd()
    if cwd:
        os.chdir(cwd)
    print(f"Executing: {command} in {os.getcwd()}")
    os.system(command)
    if cwd:
        os.chdir(original_cwd)  # Change back to original directory


def _generate_and_run_slm(
    tidal, parametric, mseos, params, eos_gen_cmd, slm_file_name_format
):
    """
    Helper function to generalize the EOS generation and SLM execution.
    """
    current_dir = os.getcwd()  # Store current directory

    # Generate EOS
    print("params:", params)
    _run_system_command(eos_gen_cmd.format(*params), cwd=EOS_CODES_DIR)

    # Construct file name and run SLM
    file_name = slm_file_name_format.format(*params)
    _run_system_command(
        f"python SLM.py {file_name} {tidal} {parametric} {mseos} {None}", cwd=SRC_DIR
    )
    os.chdir(current_dir)  # Ensure we are back in the original directory


def eval_parametric(svdSize=8, EOS_PATH=None, tidal=False, mseos=False):
    parametric = True

    # Ensure EOS directory exists
    target_eos_path = MSEOS_PATH if mseos else QEOS_PATH
    os.makedirs(target_eos_path, exist_ok=True)  # Use exist_ok=True

    if mseos is True:
        print("Generating MSEOS files and running SLM...")
        for ls in Ls:
            for lv in Lv:
                for zeta in zetaVal:
                    for xi in xiVal:
                        _generate_and_run_slm(
                            tidal,
                            parametric,
                            mseos,
                            (ls, lv, zeta, xi),
                            f"python MSEOS.py {{:.4f}} {{:.3f}} {{:.4f}} {{:.1f}}",
                            f"EOS_MS_{{:.4f}}_{{:.3f}}_{{:.4f}}_{{:.1f}}.txt",
                        )
    else:
        print("Generating Quarkyonia EOS files and running SLM...")
        fileName = os.path.join(
            target_eos_path, f"EOS_Quarkyonia_{{:.2f}}_{{:.2f}}.txt"
        )  # Use target_eos_path for Quarkyonia EOS files
        for lam in lamVal:
            for kappa in kappaVal:
                _generate_and_run_slm(
                    tidal,
                    parametric,
                    mseos,
                    (kappa, lam),  # Note: Quarkyonia.py takes kappa, lam
                    f"python Quarkyonia.py {{:.2f}} {{:.2f}}",
                    fileName,
                )
    # After all SLM runs, execute pSLM.py
    _run_system_command(
        f"python p2SLM.py {'--tidal' if tidal else ''} {'--mseos' if mseos else ''} --error_threshold 1e-10 --max_r 14 --distance_threshold 0.5 --k 4 --num_boundary_train 10 --num_train_total 20",
        cwd=SRC_DIR,
    )


def main(parametric=False, tidal=False, mseos=False):
    # Consolidate directory creation
    required_paths = [
        RESULTS_PATH,
        EOS_FILES_DIR,
        TOV_PATH,
        PLOTS_PATH,
        (
            SLM_RES_MSEOS if mseos else SLM_RES_QEOS
        ),  # Only create the relevant SLM_RES path
    ]
    for path in required_paths:
        os.makedirs(path, exist_ok=True)

    if parametric is True:
        svdSize = 14  # Default SVD size for parametric DMD
        # EOS_PATH is now handled within eval_parametric or implicitly by _generate_and_run_slm
        eval_parametric(svdSize, None, tidal, mseos)
    else:
        fileName = input("Enter the EOS file name: ")
        print(f"Running SLM for {fileName}")
        _run_system_command(
            f"python SLM.py {fileName} {tidal} {parametric} {mseos}", cwd=SRC_DIR
        )


def _get_boolean_input(prompt, default_value):
    """Helper function to handle boolean user input with robust parsing."""
    user_input = input(f"{prompt} (True/False)[{default_value}]: ").strip().lower()
    if not user_input:
        return default_value
    return user_input in ("true", "t", "1")


if __name__ == "__main__":
    cleanup = _get_boolean_input("Clean up the directories?", default_value=False)
    if cleanup is True:
        clean_directory()

    # Consolidate main directory creation for non-parametric as well
    os.makedirs(RESULTS_PATH, exist_ok=True)  # Always ensure RESULTS_PATH exists

    parametric = _get_boolean_input("Parametric or General DMD?", default_value=True)
    tidal = _get_boolean_input("Tidal or Non-Tidal?", default_value=True)

    mseos = False  # Default value
    if parametric is True:
        mseos = _get_boolean_input("MSEOS or Quarkyonia?", default_value=True)

    main(parametric, tidal, mseos)
