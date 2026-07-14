###########################################
# Parametric SLM code for TOV data
# Author: Sudhanva Lalit
# Last edited: 24 November 2024
# Updated: Using SLM.py imports
###########################################

import numpy as np
import os
import sys
import time
from scipy.spatial import distance
import json
import shutil
from itertools import combinations_with_replacement
from sklearn.neighbors import NearestNeighbors
import random
# Assuming plotData is in a relative path
from slmemulator import plot_parametric

# --- Updated Imports for Project Structure ---
# Use relative imports assuming pSLM.py is run from the project root or 
# is within a package structure.
try:
    from .SLM import augment_data_multiple_columns  # Import the function from SLM.py
    from .config import get_paths  # Import the get_paths function
except ImportError:
    # Fallback for running pSLM.py directly
    print("Warning: Running pSLM.py outside of package. Attempting direct file imports.")
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.SLM import augment_data_multiple_columns
    from src.config import get_paths
# ---------------------------------------------


# The previous path definitions are replaced by calling get_paths()
# These will be dynamically loaded in main()

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


class ParametricSLM:

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
        r"""
        Encode complex numbers as dictionaries for JSON serialization.
        """
        if isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        raise TypeError("Type not serializable")

    def sort_and_fix_eigenpairs(self, eigenvalues, eigenvectors):
        r"""
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
        r"""
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
        self,
        x0,
        y,
        x_data,
        tol=1e-8,
        max_iter=100,
        num_neighbors=5,
        reg=1e-5,
        sigma=1.0,
        max_basis_size=10,
    ):
        r"""
        Perform Banach GRIM interpolation for multidimensional data with refinements.

        Parameters:
        x0 (array-like): Initial guess (2D array) for the input.
        y (array-like): Target values (output data points).
        x_data (array-like): Input data points corresponding to the output.
        tol (float): Tolerance level to achieve.
        max_iter (int): Maximum number of iterations.
        num_neighbors (int): Number of nearest neighbors to consider for interpolation.
        reg (float): Regularization parameter to stabilize interpolation.
        sigma (float): Scale parameter for Gaussian weighting.
        max_basis_size (int): Maximum size of the basis pool for interpolation.

        Returns:
        y_interp (float): Interpolated output value for input `x0`.
        """
        x = np.array(x0)

        for i in range(max_iter):
            # Compute distances to all points in x_data
            distances = distance.cdist([x], x_data, metric="euclidean").flatten()

            # Dynamically adjust the number of neighbors based on the iteration
            neighbors = min(num_neighbors + i, max_basis_size)

            # Select the indices of the closest neighbors
            nearest_indices = np.argsort(distances)[:neighbors]
            nearest_distances = distances[nearest_indices]
            nearest_y = y[nearest_indices]
            nearest_x = x_data[nearest_indices]

            # Compute Gaussian weights with regularization
            weights = np.exp(-(nearest_distances**2) / (2 * sigma**2)) + reg
            weights /= np.sum(weights)  # Normalize weights

            # Ensure weights have the correct shape for broadcasting
            weights = weights[:, np.newaxis]

            # Interpolated y value
            y_interp = np.sum(weights * nearest_y, axis=0)

            # Refine the x point based on weighted interpolation of neighbors
            x_new = np.sum(weights * nearest_x, axis=0)

            # Compute error (norm difference between current and new point)
            error = np.linalg.norm(x_new - x)

            # Check for convergence
            if error < tol:
                return y_interp

            # Update x for the next iteration
            x = x_new

        return y_interp

    # This function is now imported from SLM.py
    # def augment_data_multiple_columns(self, X):
    #     r"""
    #     Augment the data matrix X with nonlinear terms for multiple variables.
    #     ... (original implementation)
    #     """
    #     # Not needed here since it's imported globally

    # Numpy based Reduced eigen-pair interpolation
    def fit(self):
        r"""
        Fit the data using the parametric SLM algorithm.
        """
        params = []

        # Read data from the dmd files in dmdRes
        for file in self.fileList:
            print(f"fileName = {file}")
            nameList = file.strip("MR.dat").split("_")
            params.append(nameList[2:])
            data = np.loadtxt(self.filePath + "/" + file).T
            print("Data", data.shape)
            self.t = np.arange(len(data.T[0]))
            self.dt = (self.t[-1] - self.t[0]) / len(self.t)

            # newData = np.delete(data, 0, axis=1)
            X = [np.log(data[0]), np.log(data[1]), np.log(data[2])]
            if self.tidal:
                X.append(np.log(data[3]))
            X = np.array(X, dtype=np.float64)

            self.n = len(X)
            # --- Using imported augment_data_multiple_columns ---
            X = augment_data_multiple_columns(X)
            # ----------------------------------------------------
            X1 = np.delete(X, -1, axis=1)
            X2 = np.delete(X, 0, axis=1)
            self.mm1 = X1.shape[1] + 1

            # Compute SVD of X1
            U, S, Vt = np.linalg.svd(X1, full_matrices=False)

            # Truncate to rank r
            self.r = min(self.svdSize, U.shape[1])
            U_r = U[:, : self.r]
            S_r = np.diag(S[: self.r])
            V_r = Vt[: self.r, :]

            # Compute Atilde
            Atilde = U_r.T @ X2 @ V_r.T @ np.linalg.inv(S_r)

            # Compute eigenvectors and eigenvalues
            # D, W_r = np.linalg.eig(Atilde)
            D, W_r = self.consistent_eigen(Atilde)

            Phi = X2 @ V_r.T @ np.linalg.inv(S_r) @ W_r  # DMD modes
            # Phi = U_r @ W_r

            # Compute DMD mode amplitudes b
            x1 = X1[:, 0]
            # b = np.linalg.lstsq(Phi, x1, rcond=None)[0]
            b = self.regularized_lstsq(Phi, x1, alpha=1e-8)
            # b, residuals, rank, s = lstsq(Phi, x1)

            self.LamrVals.append(D)
            self.UrVals.append(U_r)
            self.WrVals.append(W_r)
            self.VrVals.append(V_r)
            self.SrVals.append(S_r)
            self.bVals.append(b)
            self.AtildeVals.append(Atilde)

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
        
        # --- Use dynamic path for TRAIN_PATH ---
        paths = get_paths()
        train_path = paths["train_path"]
        
        if not os.path.exists(train_path):
            os.makedirs(train_path)
            
        serializable_data = json.dumps(
            dArray, default=self.complex_encoder, sort_keys=True, indent=4
        )
        with open(f"{train_path}/dmdData.json", "w") as f:
            f.write(serializable_data)
        # ---------------------------------------

    def predict(self, theta):
        r"""
        Predict the output for a given input parameter theta.
        """
        # Compute phi
        theta = np.asarray([theta], dtype=np.float64)[0]
        print("theta:", theta, theta.shape)

        # Interpolate lambda
        lambda_real = self.banach_grim_data(theta, self.LamrVals.real, self.params)
        lambda_imag = self.banach_grim_data(theta, self.LamrVals.imag, self.params)
        print("lambda_real:", lambda_real)
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
        r"""
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

def split_and_copy_files(all_files, source_dir, train_dest, val_dest, test_dest, ratios):
    """Randomly splits files and copies them to T/V/T directories."""
    
    random.shuffle(all_files)
    total_files = len(all_files)
    
    # Calculate split indices
    train_end = int(total_files * ratios[0])
    val_end = train_end + int(total_files * ratios[1])
    
    # Split the file list
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]
    
    print(f"\nTotal files: {total_files}")
    print(f"Train files: {len(train_files)} ({ratios[0]*100}%)")
    print(f"Validation files: {len(val_files)} ({ratios[1]*100}%)")
    print(f"Test files: {len(test_files)} ({ratios[2]*100}%)")

    # Copy files
    print("\nStarting file copy...")
    copy_tasks = [
        (train_files, train_dest),
        (val_files, val_dest),
        (test_files, test_dest)
    ]
    
    for file_list, dest_dir in copy_tasks:
        clean_and_create_dir(dest_dir)
        for filename in file_list:
            source_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            shutil.copy2(source_file, dest_file) # Use copy2 to preserve metadata
        print(f"Copied {len(file_list)} files to {dest_dir}")
        
    return train_files, test_files


def main(tidal=False, mseos=False):
    # --- Dynamic Path Loading ---
    paths = get_paths()
    
    # Define source and destination paths based on EOS type
    if mseos:
        tov_data_source = paths["mseos_tov_path"] # TOV_data/MSEOS/
        train_tov_dest = paths["train_tov_path"]  # trainData/MSEOS/
        val_tov_dest = paths["val_tov_path"]      # valData/MSEOS/
        test_tov_dest = paths["test_tov_path"]    # testData/MSEOS/
    else:
        tov_data_source = paths["qeos_tov_path"] # TOV_data/QEOS/
        train_tov_dest = paths["q_train_tov_path"]
        val_tov_dest = paths["q_val_tov_path"]
        test_tov_dest = paths["q_test_tov_path"]
        
    # Standard T/V/T split ratios (e.g., 70/15/15)
    ratios = [0.70, 0.15, 0.15]
    random.seed(48824) # Set seed for reproducibility

    # 1. Get all relevant files (MR_*.txt/dat)
    all_files = [f for f in os.listdir(tov_data_source) if "MR" in f]
    
    if not all_files:
        print(f"Error: No 'MR' files found in {tov_data_source}. Exiting.")
        return

    # 2. Split and Copy Files to T/V/T directories
    train_files, test_files = split_and_copy_files(
        all_files, tov_data_source, train_tov_dest, val_tov_dest, test_tov_dest, ratios
    )

    # 3. Fit on the Training Data
    # NOTE: ParametricSLM.__init__ is updated to receive the list of training filenames
    # and the original source path (for context, though files are loaded from train_tov_dest in fit).
    training = ParametricSLM(train_files, tov_data_source, 6, tidal)
    training.fit()

    # 4. Predict/Evaluate on the Test Data
    # The os.chdir is simplified. We use os.path.join for all paths now.
    print(f"\nEvaluating on {len(test_files)} test files from {test_tov_dest}")
    time_dict = {}

    for file in test_files: # Iterate only over the list of test files
        if "MR" in file:
            fileSplit = file.strip(".dat").split("_")
            testParam = fileSplit[2:]
            
            starttime = time.time()
            phi, omega, lam, b, Xdmd, newT = training.predict(testParam)
            stoptime = time.time()
            
            # Load the original data from the test folder
            data = np.loadtxt(os.path.join(test_tov_dest, file))
            X = data.T
            name = "Data_" + "_".join(testParam)
            
            plot_parametric(Xdmd, X, name, tidal)
            print(f"plotted file: {name}")
            total_time = stoptime - starttime
            time_dict[file] = total_time

    serializable_data = json.dumps(time_dict, sort_keys=True, indent=4)
    
    # Save the time data to the base test path directory
    with open(os.path.join(paths["test_data_path"], "dmd_time_data.json"), "w") as f:
        f.write(serializable_data)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pSLM.py <tidal_bool> <mseos_bool>")
        sys.exit(1)
        
    # Use standard evaluation for bools
    try:
        tidal = sys.argv[1].lower() in ('true', '1', 't')
        mseos = sys.argv[2].lower() in ('true', '1', 't')
    except:
        print("Invalid arguments. tidal and mseos must be boolean (True/False).")
        sys.exit(1)
        
    main(tidal, mseos)