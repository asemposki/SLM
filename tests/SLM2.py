import numpy as np
from scipy.linalg import logm  # logm is in scipy.linalg, not numpy.linalg


def calculate_A_with_DMD(file_path, rk4_time_step, svd_rank=None):
    """
    Reads RK4-generated solution data from a file, extracts relevant state columns,
    splits it into X1 and X2, and calculates the discrete-time propagation matrix A
    using the Dynamic Mode Decomposition (DMD) technique.

    Args:
        file_path (str): The path to the file containing the RK4 solution data.
                         Assumes data is space-separated with columns r, p, m, y.
        rk4_time_step (float): The time step (dr) used in the RK4 simulation.
        svd_rank (int, optional): The truncation rank for SVD. If None, no truncation
                                  is applied (full rank SVD is used). This can help
                                  denoise data or reduce dimensionality.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The calculated matrix A (DMD operator).
            - numpy.ndarray: The X1 matrix used for calculation.
            - numpy.ndarray: The X2 matrix used for calculation.
            - numpy.ndarray: The estimated continuous-time matrix F.
    """
    try:
        data = np.loadtxt(file_path)
        print(f"Successfully loaded data from {file_path}. Shape: {data.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

    # The state vector x is [p, m, y]
    # In the provided data:
    # Column 0 is r
    # Column 1 is p
    # Column 2 is m
    # Column 3 is y
    X = data[:, 1:4]  # Select columns p, m, y as the state
    print(f"Extracted state data (p, m, y) X. Shape: {X.shape}")

    n_snapshots, state_dimension = X.shape

    if n_snapshots < 2:
        print("Error: Not enough snapshots to create X1 and X2. Need at least 2.")
        return None, None, None, None

    # X1 has snapshots 1 through n-1 (Python indexing: 0 to n-2)
    # X1 will have n-1 rows and d columns
    X1 = X[:-1, :].T  # Transpose to have state vectors as columns (d x (n-1))

    # X2 has snapshots 2 through n (Python indexing: 1 to n-1)
    # X2 will have n-1 rows and d columns
    X2 = X[1:, :].T  # Transpose to have state vectors as columns (d x (n-1))

    print(f"Shape of X1 (transposed, columns are state vectors): {X1.shape}")
    print(f"Shape of X2 (transposed, columns are state vectors): {X2.shape}")

    # --- DMD Calculation of A ---
    # 1. Perform Singular Value Decomposition on X1
    # U: Left singular vectors, s: singular values, Vh: Right singular vectors (transposed)
    U, s, Vh = np.linalg.svd(X1, full_matrices=False)

    # Apply SVD truncation if a rank is specified
    if svd_rank is not None and svd_rank < len(s):
        U = U[:, :svd_rank]
        s = s[:svd_rank]
        Vh = Vh[:svd_rank, :]
        print(f"SVD truncated to rank: {svd_rank}")
    else:
        print("Using full rank SVD (no truncation).")

    # Compute the pseudo-inverse of Sigma (from s)
    Sigma_inv = np.diag(1.0 / s)

    # Compute the DMD operator (matrix A)
    # A = X2 @ V @ Sigma_inv @ U.T
    # Note: Vh is V.T, so V is Vh.T
    try:
        A_dmd = X2 @ Vh.T @ Sigma_inv @ U.T
        print(f"Successfully calculated matrix A using DMD. Shape: {A_dmd.shape}")
    except np.linalg.LinAlgError as e:
        print(f"Error calculating DMD operator A: {e}")
        print(
            "This might happen if singular values are too small or X1 is highly degenerate."
        )
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during DMD A calculation: {e}")
        return None, None, None, None

    # Estimate F if the time step is known
    F_estimated = None
    if A_dmd is not None and rk4_time_step is not None and rk4_time_step != 0:
        try:
            # Using scipy.linalg.logm for matrix logarithm
            F_estimated = (1 / rk4_time_step) * logm(A_dmd)
            print(f"Estimated continuous-time matrix F. Shape: {F_estimated.shape}")
        except np.linalg.LinAlgError as e:
            print(f"Error calculating matrix logarithm for F: {e}")
            print(
                "This can occur if A has negative or complex eigenvalues, or is singular."
            )
        except Exception as e:
            print(f"An unexpected error occurred during F estimation: {e}")

    return A_dmd, X1, X2, F_estimated


# --- Configuration ---
data_file_name = "MR_Quarkyonia_0.10_300.00.txt"

# Determine the time step (dr) from the r column in the data
try:
    r_values = np.loadtxt(data_file_name)[:, 0]  # r is the first column
    rk4_time_step = np.diff(r_values)[0]
    print(f"Detected time step (dr) from data: {rk4_time_step}")
except Exception as e:
    print(
        f"Could not determine time step from file: {e}. Please provide it manually if needed for F estimation."
    )
    rk4_time_step = None

# Optional: Set SVD truncation rank.
# If None, full rank SVD is used.
# If an integer, the SVD will be truncated to that rank.
svd_truncation_rank = (
    None  # e.g., 3 for a 3-dimensional system, or None for no truncation.
)

# --- Main Calculation ---
A_dmd_calculated, X1_matrix, X2_matrix, F_estimated_dmd = calculate_A_with_DMD(
    data_file_name, rk4_time_step, svd_rank=svd_truncation_rank
)

if A_dmd_calculated is not None:
    print("\nCalculated Matrix A using DMD (such that X2 = A @ X1):")
    print(A_dmd_calculated)

    # Verification: X2 should be approximately A @ X1
    if X1_matrix is not None and X2_matrix is not None:
        X2_predicted = A_dmd_calculated @ X1_matrix
        print(
            "\nVerification: Comparison of X2_predicted (A @ X1) and actual X2 (first 5 state vectors):"
        )
        print("X2_predicted (first 5):\n", X2_predicted[:, :5])
        print("X2_matrix (first 5):\n", X2_matrix[:, :5])
        # Calculate the Frobenius norm of the difference
        difference_norm = np.linalg.norm(X2_predicted - X2_matrix, "fro")
        print(
            f"\nFrobenius Norm of difference ||X2_predicted - X2_matrix||: {difference_norm}"
        )

    if F_estimated_dmd is not None:
        print("\nEstimated Continuous-Time Matrix F (from DMD A and time step):")
        print(F_estimated_dmd)

    # Plot predicted vs actual X2 for visual verification
    import matplotlib.pyplot as plt

    if X1_matrix is not None and X2_matrix is not None:
        plt.figure(figsize=(12, 6))
        for i in range(1, 2):  # Plot each state variable
            plt.plot(r_values[1:], X2_matrix[i, :], label=f"Actual X2 State {i+1}")
            plt.plot(
                r_values[1:],
                X2_predicted[i, :],
                "*",
                label=f"Predicted X2 State {i+1}",
            )
        plt.title("Comparison of Actual vs Predicted X2 States")
        plt.xlabel("Snapshot Index")
        plt.ylabel("State Value")
        plt.legend()
        plt.grid()
        plt.show()
