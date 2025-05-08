"""Script to evaluate DMDs for the TOV equations
Author: Sudhanva Lalit
Last edited: 23 September 2024
"""

import numpy as np
import os
import sys
import time
from scipy.spatial import distance
import json
from itertools import combinations_with_replacement
from sklearn.neighbors import NearestNeighbors
import random
from plotData import plot_parametric, plot_eigs

BASE_PATH = os.path.join(os.path.dirname(__file__), "..")
# DMD_DATA_PATH = f"{BASE_PATH}/Results/"
TEST_DATA_PATH = f"{BASE_PATH}/testData/"
TOV_DATA_PATH = f"{BASE_PATH}/TOV_data/"
TRAIN_PATH = f"{BASE_PATH}/trainData/"

# Interpolation using banach grim

colors = [
    "r",
    "b",
    "g",
    "k",
    "orange",
    "purple",
    "y",
    "m",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


class ParametricDMD:

    def __init__(self, fileList, filePath, svdSize, tidal=False) -> None:
        self.fileList = fileList
        self.filePath = filePath
        self.svdSize = svdSize
        self.tidal = tidal
        self.LamrVals = []
        self.UrVals = []
        self.WrVals = []
        self.VrVals = []
        self.SrVals = []
        self.param1 = []
        self.bVals = []
        self.AtildeVals = []

    @staticmethod
    def complex_encoder(obj):
        if isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        raise TypeError("Type not serializable")

    def sort_and_fix_eigenpairs(self, eigenvalues, eigenvectors):
        """
        Sort eigenvalues and corresponding eigenvectors, and fix the sign of eigenvectors for consistency.

        Parameters:
        - eigenvalues: Array of eigenvalues.
        - eigenvectors: Array of corresponding eigenvectors.

        Returns:
        - sorted_eigenvalues: Sorted eigenvalues.
        - sorted_eigenvectors: Eigenvectors corresponding to the sorted eigenvalues, with consistent signs.
        """
        # Sort eigenvalues and eigenvectors based on the magnitude of the eigenvalues
        indices = np.argsort(np.abs(eigenvalues))
        sorted_eigenvalues = eigenvalues[indices]
        sorted_eigenvectors = eigenvectors[:, indices]

        # Fix the sign of eigenvectors for consistency
        for i in range(sorted_eigenvectors.shape[1]):
            if np.real(sorted_eigenvectors[0, i]) < 0:
                sorted_eigenvectors[:, i] = -sorted_eigenvectors[:, i]

        return sorted_eigenvalues, sorted_eigenvectors

    def consistent_eigen(self, matrix):
        """
        Compute consistent eigenvalues and eigenvectors for a given matrix.

        Parameters:
        - matrix: The input matrix for which eigenvalues and eigenvectors are computed.

        Returns:
        - sorted_eigenvalues: Sorted eigenvalues with consistent order.
        - sorted_eigenvectors: Sorted eigenvectors with consistent signs.
        """
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        # Sort eigenvalues and eigenvectors, and fix the sign of eigenvectors
        sorted_eigenvalues, sorted_eigenvectors = self.sort_and_fix_eigenpairs(
            eigenvalues, eigenvectors
        )

        return sorted_eigenvalues, sorted_eigenvectors

    # Function for the Banach GRIM interpolation algorithm
    def banach_grim_data(
        self, x0, y, x_data, tol=1e-8, max_iter=100, extrapolate=False
    ):
        """
        Perform Banach GRIM interpolation for multidimensional data.

        Parameters:
        x0 (array-like): Initial guess (2D array) for the input.
        y (array-like): Target values (output data points).
        x_data (array-like): Input data points corresponding to the output.
        tol (float): Tolerance level to achieve.
        max_iter (int): Maximum number of iterations.

        Returns:
        y_interp (float): Interpolated output value for input `x0`.
        """
        x = np.array(x0)
        for i in range(max_iter):
            # Find the nearest neighbors in the dataset based on distance
            distances = distance.cdist([x], x_data, metric="euclidean")
            nearest_idx = np.argmin(distances)

            # Interpolation step: adjust based on the closest point
            y_closest = y[nearest_idx]
            x_new = x_data[nearest_idx]

            if np.linalg.norm(x_new - x) < tol:
                print(f"Converged after {i+1} iterations.")
                return y_closest

            # Update the point to the closest one for the next iteration
            x = x_new

        print("Did not converge within the maximum number of iterations.")
        return y_closest

    def augment_data_multiple_columns(self, X):
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

    # Numpy based Reduced eigen-pair interpolation
    def fit(self):
        params = []

        # Read data from the dmd files in dmdRes
        for file in self.fileList:
            print(f"fileName = {file}")
            nameList = file.strip("MR.dat").split("_")
            params.append(nameList[2:])
            data = np.loadtxt(self.filePath + file).T
            print("Data", data.shape)
            self.t = np.arange(len(data.T[0]))
            self.dt = (self.t[-1] - self.t[0]) / len(self.t)

            # newData = np.delete(data, 0, axis=1)
            X = [np.log(data[0]), np.log(data[1]), np.log(data[2])]
            if self.tidal:
                X.append(np.log(data[3]))
            X = np.array(X, dtype=np.float64)

            self.n = len(X)
            X = self.augment_data_multiple_columns(X)
            X1 = np.delete(X, -1, axis=1)
            X2 = np.delete(X, 0, axis=1)
            self.mm1 = X1.shape[1] + 1

            # Compute SVD of X1
            U, S, Vt = np.linalg.svd(X1, full_matrices=False)
            # sorted_indices = np.argsort(-S)
            # S = S[sorted_indices]
            # U = U[:, sorted_indices]
            # Vt = Vt[sorted_indices, :]
            if (np.inf in U) or (np.inf in S) or (np.inf in Vt):
                print("\n\n!! ! !!")
            if (np.nan in U) or (np.nan in S) or (np.nan in Vt):
                print("\n\n! !!! !")

            # Truncate to rank r
            self.r = min(self.svdSize, U.shape[1])
            U_r = U[:, : self.r]
            S_r = np.diag(S[: self.r])
            V_r = Vt[: self.r, :]

            # Compute Atilde
            Atilde = U_r.T @ X2 @ V_r.T @ np.linalg.inv(S_r)

            if np.inf in Atilde:
                print("\n" * 5)
                print("inf issue !!!")
            if np.nan in Atilde:
                print("\n" * 5)
                print("nan issue !!!")

            # Compute eigenvectors and eigenvalues
            # D, W_r = np.linalg.eig(Atilde)
            D, W_r = self.consistent_eigen(Atilde)
            # print("D", W_r)

            Phi = X2 @ V_r.T @ np.linalg.inv(S_r) @ W_r  # DMD modes
            # Phi = U_r @ W_r

            # Compute DMD mode amplitudes b
            x1 = X1[:, 0]
            # print("Shapes", Phi, "\n", x1)
            # b = np.linalg.lstsq(Phi, x1, rcond=None)[0]
            b = self.regularized_lstsq(Phi, x1, alpha=1e-8)
            print("b", b)
            # b, residuals, rank, s = lstsq(Phi, x1)

            self.LamrVals.append(D)
            self.UrVals.append(U_r)
            self.WrVals.append(W_r)
            self.VrVals.append(V_r)
            self.SrVals.append(S_r)
            self.bVals.append(b)
            self.AtildeVals.append(Atilde)

        print(params)
        self.params = np.asarray(params, dtype=np.float64)
        self.LamrVals = np.array(self.LamrVals)
        self.UrVals = np.array(self.UrVals)
        self.SrVals = np.asarray(self.SrVals)
        self.WrVals = np.array(self.WrVals)
        self.VrVals = np.asarray(self.VrVals)
        self.bVals = np.array(self.bVals)
        self.AtildeVals = np.array(self.AtildeVals)

        # Reshape arrays for RBF process
        xShape = self.params.shape[0]
        self.UrValsReshape = self.UrVals.reshape(xShape, -1)
        self.WrValsReshape = self.WrVals.reshape(xShape, -1)
        self.SrVals = self.SrVals.reshape(xShape, -1)
        self.VrVals = self.VrVals.reshape(xShape, -1)
        self.bVals_real = self.bVals.real.reshape(xShape, -1)
        self.bVals_cplx = self.bVals.imag.reshape(xShape, -1)
        self.AtildeVals = self.AtildeVals.reshape(xShape, -1)

        # Write arrays to json file
        dArray = dict()
        dArray["params"] = self.params.tolist()
        # dArray["LamrVals"] = self.LamrVals.tolist()
        dArray["UrVals"] = self.UrValsReshape.tolist()
        # dArray["WrVals"] = self.WrVals.tolist()
        dArray["VrVals"] = self.VrVals.tolist()
        dArray["SrVals"] = self.SrVals.tolist()
        # dArray["bVals"] = self.bVals.tolist()
        dArray["AtildeVals"] = self.AtildeVals.tolist()
        serializable_data = json.dumps(
            dArray, default=self.complex_encoder, sort_keys=True, indent=4
        )
        if not os.path.exists(TRAIN_PATH):
            os.makedirs(TRAIN_PATH)
        with open(f"{TRAIN_PATH}/dmdData.json", "w") as f:
            f.write(serializable_data)

    def predict(self, theta):
        # Compute phi
        theta = np.asarray([theta], dtype=np.float64)[0]
        print("theta:", theta, theta.shape)

        # Interpolate lambda
        lambda_real = self.banach_grim_data(theta, self.LamrVals.real, self.params)
        lambda_imag = self.banach_grim_data(theta, self.LamrVals.imag, self.params)
        lambda_vals = lambda_real + 1j * lambda_imag
        print("lambda:", lambda_vals)

        # Interpolate Ur and Wr
        U_r = self.banach_grim_data(theta, self.UrValsReshape, self.params)
        W_r = self.banach_grim_data(theta, self.WrValsReshape, self.params)

        # Interpolate Atilde, b
        Atilde = self.banach_grim_data(theta, self.AtildeVals, self.params)
        b_real = self.banach_grim_data(theta, self.bVals_real, self.params)
        b_cplx = self.banach_grim_data(theta, self.bVals_cplx, self.params)

        WrShape = int(np.sqrt((W_r.shape[0])))
        W_r = W_r.reshape((WrShape, WrShape))

        UrLen = int(U_r.shape[0] / WrShape)
        U_r = U_r.reshape((UrLen, WrShape))

        # # Compute eigenvectors and eigenvalues
        Atilde = Atilde.reshape((WrShape, WrShape))
        # # D, W_r = np.linalg.eig(Atilde)
        D, W_r = self.consistent_eigen(Atilde)
        # print(W_r_new - W_r)

        Phi = U_r @ W_r
        # print(Phi - Phi_Interp)

        # omega = []
        omega = np.log(lambda_vals) / self.dt
        print(f"Omega: {omega}")

        # # Compute DMD mode amplitudes b
        bVals = b_real + 1j * b_cplx
        print(f"b = {bVals}")

        # DMD reconstruction
        time_dynamics = np.zeros((self.r, self.mm1), dtype=np.complex64)
        t = np.arange(self.mm1) * self.dt  # time vector
        print("time_dynanmics", time_dynamics.shape, self.r)

        for iter in range(self.mm1):
            time_dynamics[:, iter] = bVals * np.exp(omega * t[iter])

        Xdmd2 = Phi @ time_dynamics
        # print("Xdmd2", Xdmd2)
        Xdmd = Xdmd2[: self.n, :]
        Phi = Phi[: self.n, :]
        return Phi, omega, lambda_vals, bVals, Xdmd, t

    def regularized_lstsq(self, A, B, alpha=1e-8):
        """
        Solve the least squares problem with regularization (Ridge regression).

        Parameters:
        - A: Matrix of coefficients.
        - B: Right-hand side vector.
        - alpha: Regularization parameter.

        Returns:
        - x: Solution vector.
        """
        # Check condition
        cond_num = np.linalg.cond(A)
        print("Condition Number:", cond_num)

        if cond_num > 1e10:
            print("Matrix is ill-conditioned. Applying regularization.")
            # Regularization term (identity matrix scaled by alpha)
            I = np.eye(A.shape[1])
            A_reg = A.T @ A + alpha * I
            B_reg = A.T @ B

            # Solve the modified least squares problem
            x = np.linalg.solve(A_reg, B_reg)
            return x
        # Regularization term (identity matrix scaled by alpha)
        I = np.eye(A.shape[1])
        A_reg = A.T @ A + alpha * I
        B_reg = A.T @ B

        # Solve the modified least squares problem
        x = np.linalg.solve(A_reg, B_reg)
        return x


def main(tidal=False, mseos=False):
    fileList = []
    if mseos:
        tov_data_path = f"{TOV_DATA_PATH}/MSEOS/"
    else:
        print(f"{TOV_DATA_PATH}/Quarkies/")
        tov_data_path = f"{TOV_DATA_PATH}/Quarkies/"

    for file in os.listdir(tov_data_path):
        fileList.append(file)

    # random.seed(48824)
    # updatedFileList = random.sample(fileList, 10)
    # updatedFileList = sorted(updatedFileList)
    # testFileList = random.sample(fileList, 5)  # originally 10
    # testFileList = sorted(testFileList)
    # print(updatedFileList)
    # print(testFileList)
    
    # make parameter space as according to `SLM/plot_code/SLM_parametric_eos_plot.ipynb`
    random.seed(48824)
    fileList = sorted(fileList)
    updatedFileList = [file for file in fileList[::3]]
    testFileList = [file for file in fileList[::4] if file not in updatedFileList]
    updatedFileList = updatedFileList[::4]
    testFileList = testFileList[::4]
    
    print("")
    print(updatedFileList)
    print(testFileList)

    if not os.path.exists(TEST_DATA_PATH):
        os.makedirs(TEST_DATA_PATH)
    os.chdir(TEST_DATA_PATH)
    os.system("rm * && cd ../")
    for file in testFileList:
        if file not in updatedFileList:
            dmdFile = os.path.join(tov_data_path, file)
            os.system(f"cp {dmdFile} {TEST_DATA_PATH}")

    training = ParametricDMD(updatedFileList, tov_data_path, 12, tidal)  # 13
    training.fit()

    time_dict = {}

    for file in os.listdir(TEST_DATA_PATH):
        if "MR" in file:
            fileSplit = file.strip(".dat").split("_")
            testParam = fileSplit[2:]
            starttime = time.time()
            phi, omega, lam, b, Xdmd, newT = training.predict(testParam)
            stoptime = time.time()
            # plot_eigs(lam, dpi=600, filename=f"eigsplot_{file}.pdf")
            data = np.loadtxt(os.path.join(TEST_DATA_PATH, file))
            X = data.T
            name = "Data_" + "_".join(testParam)
            plot_parametric(Xdmd, X, name, tidal)
            print(f"plotted file: {name}")
            total_time = stoptime - starttime
            time_dict[file] = total_time
            
            # write the SLM data
            np.savetxt("SLM_Quarkyonia_" + "_".join(testParam) + '.dat', np.exp(Xdmd.real.T))

    serializable_data = json.dumps(time_dict, sort_keys=True, indent=4)

    with open(f"dmd_time_data.json", "w") as file:
        file.write(serializable_data)


if __name__ == "__main__":
    (tidal, mseos) = sys.argv[1:]
    tidal = eval(tidal)
    mseos = eval(mseos)
    main(tidal, mseos)
