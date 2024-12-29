###########################################
# SLM code for the TOV data
# Author: Sudhanva Lalit
# Last edited: 24 November 2024
###########################################

import numpy as np
import os
import shutil
import sys
import json
from itertools import combinations_with_replacement
import time
from TOV_class import TOVsolver
from plotData import plot_eigs, plot_S, plot_dmd, plot_dmd_rad

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
    QEOS_TOV_PATH,
    MSEOS_TOV_PATH,
    SLM_RES_MSEOS,
    SLM_RES_QEOS,
)

p0 = 1.285e3


def augment_data_multiple_columns(X):
    r"""
    Augment the data matrix X with nonlinear terms for multiple variables.

    Parameters:
    X : np.ndarray
        The data matrix where each row is a variable, and each column is a snapshot in time.

    Returns:
    augmented_X : np.ndarray
        The augmented data matrix with nonlinear terms.
    """
    # Number of variables (rows) and time snapshots (columns)
    n_variables, n_snapshots = X.shape

    # Start with the original variables in the augmented matrix
    augmented_X = X.copy()

    # Add quadratic terms and cross-products
    for i, j in combinations_with_replacement(range(n_variables), 2):
        augmented_term = X[i, :] * X[j, :]
        augmented_X = np.vstack([augmented_X, augmented_term])

    return augmented_X


# Numpy based DMD
def DMD(X, r, dt):
    r"""
    Dynamic Mode decomposition for the augmented Data

    Parameters:
    X: np.ndarray
        The data matrix where each row is a variable, and each column is a snapshot in time.
    r: int
        Size of the truncated SVD
    dt: np.float
        Delta T: the time difference of linear DMDs
    """
    n = len(X)

    X = augment_data_multiple_columns(X)
    X1 = np.delete(X, -1, axis=1)
    X2 = np.delete(X, 0, axis=1)
    print(X.shape, n)

    # Compute SVD of X1
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)

    # Truncate to rank r
    r = min(r, U.shape[1])
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    V_r = Vt[:r, :]

    # Compute Atilde
    Atilde = U_r.T @ X2 @ V_r.T @ np.linalg.inv(S_r)

    # Compute eigenvectors and eigenvalues
    D, W_r = np.linalg.eig(Atilde)

    # Phi = U_r @ W_r

    Phi = X2 @ V_r.T @ np.linalg.inv(S_r) @ W_r  # DMD modes
    lambda_vals = D  # discrete-time eigenvalues
    omega = np.log(lambda_vals) / dt  # continuous-time eigenvalue

    # Compute DMD mode amplitudes b
    x1 = X1[:, 0]
    b = np.linalg.lstsq(Phi, x1, rcond=None)[0]

    # DMD reconstruction
    mm1 = X1.shape[1] + 1  # mm1 = m - 1
    time_dynamics = np.zeros((r, mm1), dtype=complex)
    t = np.arange(mm1) * dt  # time vector

    # time dynamics upto a given time.
    for iter in range(mm1):
        time_dynamics[:, iter] = b * np.exp(omega * t[iter])

    # Finally collect the DMDs upto rank n.
    Xdmd2 = Phi @ time_dynamics
    Xdmd = Xdmd2[:n, :]
    Phi = Phi[:n, :]

    return Phi, omega, lambda_vals, b, Xdmd, S


def solve_tov(fileName, tidal=False, parametric=False, mseos=True):
    r"""
    Solves the TOV equation and returns radius, mass and central pressure

    Parameters:
        fileName (str): Filename containing the EOS in the format nb (fm^-3),
            E (MeV), P (MeV/fm^3)

    Returns:
        dataArray (array): Data array containing radii, central pressure
            and mass.
    """
    if parametric is False:
        eos_file = EOS_DATA_DIR + fileName
    else:
        if mseos is True:
            eos_file = os.path.join(MSEOS_PATH, fileName)
            TOV_PATH = MSEOS_TOV_PATH
        else:
            eos_file = os.path.join(QEOS_PATH, fileName)
            # eos_file = QEOS_PATH + "/" + fileName
            TOV_PATH = QEOS_TOV_PATH
    if not os.path.exists(TOV_PATH):
        os.makedirs(TOV_PATH)

    # Replace the filename and run the code
    file = TOVsolver(eos_file, tidal=tidal)
    file.tov_routine(verbose=False, write_to_file=False)
    print("R of 1.4 solar mass star: ", file.canonical_NS_radius())
    dataArray = [
        file.total_radius.flatten(),
        file.total_pres_central.flatten(),
        file.total_mass.flatten(),
    ]
    if tidal is True:
        dataArray.append(file.k2.flatten())
        # dataArray.append(file.tidal_deformability.flatten()[::-1])

    dataArray = np.asarray(dataArray, dtype=np.float64)
    if "_" in fileName:
        nameList = fileName.split("_")
        if len(nameList) > 2:
            file = "MR_" + "_".join(nameList[1:])
        else:
            file = "_".join(["MR", nameList[0], "TOV"])
    else:
        nameList = fileName.split(".")  # rstrip(".dat")  #.split("_")
    if len(nameList) > 2:
        file = "MR_" + "_".join(nameList[1:])
    else:
        file = "_".join(["MR", nameList[0], "TOV"])
    np.savetxt(file, dataArray.T, fmt="%1.8e")
    shutil.move(file, TOV_PATH)
    return dataArray


def main(fileName, svdSize, tidal=False, parametric=False, mseos=True):
    r"""
    Main function to run the SLM code. Solves the TOV equation and
    computes the SLM modes.

    Parameters:
        fileName (str): Filename containing the EOS in the format nb (fm^-3),
            E (MeV), P (MeV/fm^3)
        svdSize (int): Size of the truncated SVD
        tidal (bool): Whether to include tidal deformability
        parametric (bool): Whether the EOS is parametric
        mseos (bool): Whether to use MSEOS

    Returns:
        linT (np.ndarray): Time vector
        phi (np.ndarray): DMD modes
        omega (np.ndarray): Continuous-time eigenvalues
        lam (np.ndarray): Discrete-time eigenvalues
        b (np.ndarray): DMD mode amplitudes
        Xdmd (np.ndarray): DMD reconstruction
        HFTime (float): Time taken for solving TOV
        DMDTime (float): Time taken for DMD
    """
    startHFTime = time.time()
    if tidal is True:
        radius, pcentral, mass, tidal_def = solve_tov(
            fileName, tidal, parametric, mseos
        )
    else:
        radius, pcentral, mass = solve_tov(fileName, tidal, parametric, mseos)
    endHFTime = time.time()
    p = pcentral
    r = radius
    m = mass
    # print(mass)
    linT = np.arange(len(p))
    X = [np.log(r), np.log(p), np.log(m)]
    if tidal is True:
        X.append(np.log(tidal_def))

    X = np.asarray(X, dtype=np.float64)
    # print("X shape: ", X)
    startDMDTime = time.time()
    phi, omega, lam, b, Xdmd, S = DMD(X, svdSize, (linT[-1] - linT[0]) / len(linT))
    endDMDTime = time.time()
    rad_DMD = np.exp(Xdmd[0].real)
    pres_DMD = np.exp(Xdmd[1].real)
    mass_DMD = np.exp(Xdmd[2].real)

    # Make plots
    # Plot the S values
    plot_S(S)

    # Plot the eigenvalues
    plot_eigs(lam, filename="eigenValues.pdf")

    # Plot the DMDs
    fileNames = ["radiusDMD.png", "pressureDMD.png", "massDMD.png"]
    ylabels = ["Radius (km) ", r"Pressure $(MeV/fm^3)$", r"Mass $M_{(\odot)}$"]
    if tidal is True:
        fileNames.append("tidalDMD.png")
        ylabels.append(r"$\Lambda$")
    plot_dmd(linT, X, Xdmd, fileNames, ylabels, fileName)
    # Plot Mass vs Radius
    ylabels = [r"Pressure $(MeV/fm^3)$", r"Mass $M_{(\odot)}$"]
    newFiles = ["pressure_radius.png", "mass_radius.png"]
    if tidal is True:
        ylabels.append(r"$k_2$")
        newFiles.append("tidal_radius.png")
    plot_dmd_rad(X, Xdmd, newFiles, ylabels, fileName)

    # maximum values
    max_mass_DMD = np.max(mass_DMD)
    max_index = np.where([mass_DMD[i] == max_mass_DMD for i in range(len(mass_DMD))])[
        0
    ][0]
    max_radius_DMD = rad_DMD[max_index]

    print(
        "DMD Maximum mass: {}; maximum radius: {}".format(max_mass_DMD, max_radius_DMD)
    )
    HFTime = endHFTime - startHFTime
    DMDTime = endDMDTime - startDMDTime
    return linT, phi, omega, lam, b, Xdmd, HFTime, DMDTime


def complex_encoder(obj):
    r"""
    Complex encoder for JSON serialization. Converts complex numbers to
    real and imaginary parts.
    """
    if isinstance(obj, complex):
        return {"__complex__": True, "real": obj.real, "imag": obj.imag}
    raise TypeError("Type not serializable")


if __name__ == "__main__":
    argv = sys.argv
    (fileName, svdSize, tidal, parametric, mseos) = argv[1:]
    print("File name: ", fileName, parametric, mseos)
    nameList = fileName.strip(".dat").split("_")
    name = "SLM_" + "_".join(nameList[1:]) + ".dat"
    t, phi, omega, lam, b, Xdmd, HFTime, DMDTime = main(
        fileName, int(svdSize), eval(tidal), eval(parametric), eval(mseos)
    )
    if eval(parametric) is True:
        if eval(mseos) is True:
            if os.path.exists(SLM_RES_MSEOS) is False:
                os.makedirs(SLM_RES_MSEOS)
            os.chdir(SLM_RES_MSEOS)
        else:
            if os.path.exists(SLM_RES_QEOS) is False:
                os.makedirs(SLM_RES_QEOS)
            os.chdir(SLM_RES_QEOS)
    else:
        os.chdir(RESULTS_PATH)
    XdmdRes = []
    for i in range(len(Xdmd) - 1):
        XdmdRes.append(np.exp(Xdmd[i].real)[::-1])
    XdmdRes.append(np.exp(Xdmd[-1].real))
    XdmdRes = np.asarray(XdmdRes, dtype=np.float64)
    data = dict()
    data.update({"time": (t).tolist()})
    if len(nameList) > 2:
        data.update({"val": nameList[2:]})
    data.update({"phi": np.array(phi, dtype=np.complex64).tolist()})
    data.update({"omega": omega.real.astype("float64").tolist()})
    data.update({"lam": np.array(lam, dtype=np.complex64).tolist()})
    data.update({"b": b.real.astype("float64").tolist()})
    data.update({"Xdmd": (XdmdRes.real.astype("float64")).tolist()})
    data.update({"hftime": HFTime})
    data.update({"dmdtime": DMDTime})

    serializable_data = json.dumps(
        data, default=complex_encoder, sort_keys=True, indent=4
    )
    print("Path: ", os.getcwd())
    with open(f"{name}", "w") as file:
        file.write(serializable_data)
