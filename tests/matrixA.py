import numpy as np


def calculate_A_from_RK4_data(file_path, rk4_time_step):
    """
    Reads RK4-generated solution data from a file, extracts relevant state columns,
    splits it into X1 and X2, and calculates the discrete-time propagation matrix A
    such that X2 = A @ X1.

    Args:
        file_path (str): The path to the file containing the RK4 solution data.
                         Assumes data is space-separated with columns r, p, m, y.
        rk4_time_step (float): The time step (dr) used in the RK4 simulation.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The calculated matrix A.
            - numpy.ndarray: The X1 matrix used for calculation.
            - numpy.ndarray: The X2 matrix used for calculation.
            - numpy.ndarray: The estimated continuous-time matrix F.
    """
    try:
        # Load the data from the .txt file, assuming space-separated values
        # Columns are r, p, m, y
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

    # Calculate A using the pseudo-inverse: A = X2 @ X1_pseudo_inverse
    # This is equivalent to solving A @ X1 = X2 in a least-squares sense.
    try:
        A = X2 @ np.linalg.pinv(X1)
        print(f"Successfully calculated matrix A. Shape: {A.shape}")
    except np.linalg.LinAlgError as e:
        print(f"Error calculating pseudo-inverse: {e}")
        print(
            "This might happen if X1 is rank-deficient (e.g., all snapshots are identical or linearly dependent)."
        )
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during A calculation: {e}")
        return None, None, None, None

    # Estimate F if the time step is known
    F_estimated = None
    if A is not None and rk4_time_step is not None and rk4_time_step != 0:
        try:
            F_estimated = (1 / rk4_time_step) * np.linalg.logm(A)
            print(f"Estimated continuous-time matrix F. Shape: {F_estimated.shape}")
        except np.linalg.LinAlgError as e:
            print(f"Error calculating matrix logarithm for F: {e}")
            print(
                "This can occur if A has negative or complex eigenvalues, or is singular."
            )
        except Exception as e:
            print(f"An unexpected error occurred during F estimation: {e}")

    return A, X1, X2, F_estimated


# --- Configuration ---
data_file_name = "MR_Quarkyonia_0.10_300.00.txt"

# Determine the time step (dr) from the r column in the data
try:
    r_values = np.loadtxt(data_file_name)[:, 0]  # r is the first column
    # Assuming a uniform step size, take the first difference
    rk4_time_step = np.diff(r_values)[0]
    print(f"Detected time step (dr) from data: {rk4_time_step}")
except Exception as e:
    print(
        f"Could not determine time step from file: {e}. Please provide it manually if needed for F estimation."
    )
    rk4_time_step = (
        None  # Set to None if autodetection fails or you want to provide it manually
    )

# --- Main Calculation ---
A_calculated, X1_matrix, X2_matrix, F_estimated = calculate_A_from_RK4_data(
    data_file_name, rk4_time_step
)

if A_calculated is not None:
    print("\nCalculated Matrix A (such that X2 = A @ X1):")
    print(A_calculated)

    # Verify: X2 should be approximately A @ X1
    if X1_matrix is not None and X2_matrix is not None:
        X2_predicted = A_calculated @ X1_matrix
        print(
            "\nVerification: Comparison of X2_predicted (A @ X1) and actual X2 (first 5 state vectors):"
        )
        print("X2_predicted (first 5):\n", X2_predicted[:, :])
        print("X2_matrix (first 5):\n", X2_matrix[:, :])
        # Calculate the Frobenius norm of the difference
        difference_norm = np.linalg.norm(X2_predicted - X2_matrix, "fro")
        print(
            f"\nFrobenius Norm of difference ||X2_predicted - X2_matrix||: {difference_norm}"
        )

    if F_estimated is not None:
        print("\nEstimated Continuous-Time Matrix F (from A and time step):")
        print(F_estimated)

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
