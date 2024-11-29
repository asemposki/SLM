###########################################
# Test script for parametric or general DMD
# Author: Sudhanva Lalit
# Last edited: 24 November 2024
###########################################

import os
import subprocess
import sys
import numpy as np
from src.cleanData import clean_directory

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
)

# set parameters for lambda and kappa
lamVal = np.linspace(300, 500, 5)  # np.arange(300, 500, 40)  # 370 - 385
kappaVal = np.linspace(0.1, 0.3, 5)
Ls = np.linspace(0.0, 3e-3, 4)
Lv = np.linspace(0.0, 3e-2, 4)
zetaVal = np.linspace(1e-4, 2e-4, 2)
# zetaVal = [1e-4]
xiVal = np.linspace(0.0, 1.0, 2)
# xiVal = [1.0]


def eval_parametric(svdSize=8, EOS_PATH=None, tidal=False, mseos=False):
    parametric = True

    if mseos is True:
        if not os.path.exists(MSEOS_PATH):
            os.makedirs(MSEOS_PATH)
        for lam in Ls:
            for kappa in Lv:
                for zeta in zetaVal:
                    for xi in xiVal:
                        # Make MS EOS
                        os.chdir(EOS_CODES_DIR)
                        fileName = (
                            f"EOS_MS_{lam:.4f}_{kappa:.3f}_{zeta:.4f}_{xi:.1f}.dat"
                        )
                        os.system(
                            f"python MSEOS.py {lam} {kappa} {zeta} {xi} {fileName}"
                        )
                        fileName = (
                            f"EOS_MS_{lam:.4f}_{kappa:.3f}_{zeta:.4f}_{xi:.1f}.dat"
                        )
                        print(fileName)
                        os.chdir(SRC_DIR)
                        os.system(
                            f"python SLM.py {fileName} {svdSize} {tidal} {parametric} {mseos}"
                        )
    else:
        if not os.path.exists(QEOS_PATH):
            os.makedirs(QEOS_PATH)
        for lam in lamVal:
            for kappa in kappaVal:
                # Make Quarkies
                os.chdir(EOS_CODES_DIR)
                fileName = f"EOS_Quarkyonia_{lam:.2f}_{kappa:.2f}.dat"
                os.system(f"python Quarkyonia.py {kappa} {lam}")
                print(fileName)
                os.chdir(SRC_DIR)
                os.system(
                    f"python SLM.py {fileName} {svdSize} {tidal} {parametric} {mseos}"
                )

    os.system(f"python pSLM.py {tidal} {mseos}")


def main(parametric=False, tidal=False, mseos=False):
    if parametric is True:
        if mseos is True:
            EOS_PATH = MSEOS_PATH
            svdSize = 14
        else:
            EOS_PATH = QEOS_PATH
            svdSize = 10
        eval_parametric(svdSize, EOS_PATH, tidal, mseos)  # change to 8 and run from 6
    else:
        EOS_PATH = EOS_DATA_DIR
        fileName = input("Enter the EOS file name: ")
        svdSize = int(input("Enter the SVD size[Int]: "))
        print(f"Running DMD for {fileName}")
        os.system(f"python SLM.py {fileName} {svdSize} {tidal} {parametric} {mseos}")


if __name__ == "__main__":
    cleanup = input("Clean up the directories? (True/False)[False]: ") or False
    if isinstance(cleanup, str):
        cleanup = eval(cleanup)
    if cleanup is True:
        clean_directory()
    # Check if folders exist
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    if not os.path.exists(EOS_FILES_DIR):
        os.makedirs(EOS_FILES_DIR)
    if not os.path.exists(TOV_PATH):
        os.makedirs(TOV_PATH)
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)
    parametric = input("Parametric or General DMD? (True/False)[True]: ") or True
    tidal = input("Tidal or Non-Tidal? (True/False)[True]: ") or True
    if isinstance(parametric, str):
        parametric = eval(parametric)
    if isinstance(tidal, str):
        tidal = eval(tidal)
    if parametric is True:
        mseos = input("MSEOS or Quarkies? (True/False)[True]: ") or True
        if isinstance(mseos, str):
            mseos = eval(mseos)
        if mseos is True:
            if not os.path.exists(RESULTS_MSEOS_PATH):
                os.makedirs(RESULTS_MSEOS_PATH)
        else:
            if not os.path.exists(RESULTS_QUARKIES_PATH):
                os.makedirs(RESULTS_QUARKIES_PATH)
        main(parametric, tidal, mseos)
    else:
        if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)
        main(parametric, tidal)
