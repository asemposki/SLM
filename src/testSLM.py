"""
Script to test parametric or general DMD code
Author: Sudhanva Lalit
Started: 09/14/2024
Latest Update: 10/22/2024
"""

import os
import subprocess
import sys
import numpy as np


base_PATH = os.path.join(os.path.dirname(__file__), "..")
src_PATH = f"{base_PATH}/src/"
EOS_FILES_PATH = f"{base_PATH}/EOS_files/"
RESULTS_PATH = f"{base_PATH}/Results"
EOS_CODE_PATH = f"{base_PATH}/EOS_Codes/"
MR_PATH = f"{base_PATH}/TOV_data"
Quarkies_PATH = f"{base_PATH}/EOS_files/Quarkies/"
MSEOS_PATH = f"{base_PATH}/EOS_files/MSEOS/"
PLOTS_PATH = f"{base_PATH}/Plots/"
TRAIN_PATH = f"{base_PATH}/trainData/"


# set parameters for lambda and kappa
lamVal = np.linspace(300, 500, 20)  # np.arange(300, 500, 40)  # 370 - 385
kappaVal = np.linspace(0.1, 0.3, 10)
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
                        os.chdir(EOS_CODE_PATH)
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
                        os.chdir(src_PATH)
                        os.system(
                            f"python SLM.py {fileName} {svdSize} {tidal} {parametric} {mseos}"
                        )
    else:
        if not os.path.exists(Quarkies_PATH):
            os.makedirs(Quarkies_PATH)
        for lam in lamVal:
            for kappa in kappaVal:
                # Make Quarkies
                os.chdir(EOS_CODE_PATH)
                fileName = f"EOS_Quarkyonia_{lam:.2f}_{kappa:.2f}.dat"
                os.system(f"python Quarkyonia.py {kappa} {lam}")
                print(fileName)
                os.chdir(src_PATH)
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
            EOS_PATH = Quarkies_PATH
            svdSize = 10
        eval_parametric(svdSize, EOS_PATH, tidal, mseos)  # change to 8 and run from 6
    else:
        EOS_PATH = f"{base_PATH}/EOS_Data/"
        fileName = input("Enter the EOS file name: ")
        svdSize = int(input("Enter the SVD size[Int]: "))
        print(f"Running DMD for {fileName}")
        os.system(f"python SLM.py {fileName} {svdSize} {tidal} {parametric} {mseos}")


if __name__ == "__main__":
    cleanup = input("Clean up the directories? (True/False)[False]: ") or False
    if isinstance(cleanup, str):
        cleanup = eval(cleanup)
    if cleanup is True:
        os.system("python cleanData.py")
    # Check if folders exist
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    if not os.path.exists(EOS_FILES_PATH):
        os.makedirs(EOS_FILES_PATH)
    if not os.path.exists(MR_PATH):
        os.makedirs(MR_PATH)
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
        main(parametric, tidal, mseos)
    else:
        main(parametric, tidal)
