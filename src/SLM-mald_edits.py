"""Script to evaluate DMDs for the TOV equations
Author: Sudhanva Lalit
Last edited: 20 September 2024
"""

import numpy as np
import os
import sys
import json
from itertools import combinations_with_replacement
import time
from TOV_class import TOVsolver
from plotData import plot_eigs, plot_S, plot_dmd, plot_dmd_rad

BASE_PATH = os.path.join(os.path.dirname(__file__), "..")
EOS_DATA_PATH = f"{BASE_PATH}/EOS_Data/"
EOS_Quarkies = f"{BASE_PATH}/EOS_files/Quarkies/"  # /Quarkies/"
EOS_MSEOS = f"{BASE_PATH}/EOS_files/MSEOS/"
dmdResPath = f"{BASE_PATH}/Results/"
DMD_RES_MSEOS = f"{BASE_PATH}/Results/MSEOS/"
DMD_RES_Quarkies = f"{BASE_PATH}/Results/Quarkies/"
TOV_PATH = f"{BASE_PATH}/TOV_data/"
TOV_MSEOS = f"{BASE_PATH}/TOV_data/MSEOS/"
TOV_Quarkies = f"{BASE_PATH}/TOV_data/Quarkies/"
PLOTS_PATH = f"{BASE_PATH}/Plots/"

p0 = 1.285e3


def augment_data_multiple_columns(X):
    """
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
def DMD(X, dt):
    """
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
    # print(X.shape, n)

    # Compute SVD of X1
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    print("S = ", S)
    # print(U.shape)
    
    # Truncate to rank r
    """To try:
            - looking at eigenvectors for clues
    """
    time_initial = time.time()
    r_best = 0
    smallest_total_distance = 0
    for r in np.arange(1, U.shape[1] + 1):
        print(r)
        U_r = U[:, :r]
        S_r = np.diag(S[:r])
        V_r = Vt[:r, :]
        
        # Compute Atilde
        Atilde = U_r.T @ X2 @ V_r.T @ np.linalg.inv(S_r)
        
        # Compute eigenvectors and eigenvalues
        D, W_r = np.linalg.eig(Atilde)
        
        D_total = np.sum(np.abs(D.imag)) / 2
        print(f"D distance = {D_total:.6}")
        
        if D_total >= smallest_total_distance:
            r_best = r
            smallest_total_distance = D_total
        print(f"largest D distance = {smallest_total_distance:.6}")
        print(r_best)
        print("")
        
        from matplotlib import pyplot as plt
        plt.plot(W_r.real, color="blue")
        plt.plot(W_r.imag, color="orange")
        plt.show()
        
        plot_eigs(D, filename="eigenValues.pdf")
        plt.show()
        #
    time_final = time.time()
    print(f"eigenvalue search took {time_final - time_initial:.3e} seconds.")
    print(f"Using {r_best} modes.")
    
    U_r = U[:, :r_best]
    S_r = np.diag(S[:r_best])
    V_r = Vt[:r_best, :]
    
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
    time_dynamics = np.zeros((r_best, mm1), dtype=complex)
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
        eos_file = EOS_DATA_PATH + fileName
        global TOV_PATH
    else:
        if mseos is True:
            eos_file = EOS_MSEOS + fileName
            TOV_PATH = TOV_MSEOS
        else:
            eos_file = EOS_Quarkies + fileName
            TOV_PATH = TOV_Quarkies
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
    name = (fileName.split(".")[0].split("_eos")[0])
    file = "MR_" + name + ".dat"
    np.savetxt(TOV_PATH + file, dataArray.T, fmt="%1.8e")
    return dataArray


def main(fileName, tidal=False, parametric=False, mseos=True):
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
    phi, omega, lam, b, Xdmd, S = DMD(X, (linT[-1] - linT[0]) / len(linT))
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
    if isinstance(obj, complex):
        return {"__complex__": True, "real": obj.real, "imag": obj.imag}
    raise TypeError("Type not serializable")


if __name__ == "__main__":
    argv = sys.argv
    (fileName, tidal, parametric, mseos) = argv[1:]
    nameList = fileName.strip(".dat").split("_")
    name = "DMD_" + (fileName.split(".")[0].split("_eos")[0]) + ".dat"
    t, phi, omega, lam, b, Xdmd, HFTime, DMDTime = main(
        fileName, eval(tidal), eval(parametric), eval(mseos)
    )
    if mseos is True and parametric is True:
        if os.path.exists(DMD_RES_MSEOS) is False:
            os.makedirs(DMD_RES_MSEOS)
        os.chdir(DMD_RES_MSEOS)
    elif mseos is False and parametric is True:
        if os.path.exists(DMD_RES_Quarkies) is False:
            os.makedirs(DMD_RES_Quarkies)
        os.chdir(DMD_RES_Quarkies)
    else:
        os.chdir(dmdResPath)
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

    with open(f"{name}", "w") as file:
        file.write(serializable_data)
