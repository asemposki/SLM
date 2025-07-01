import numpy as np
from scipy.linalg import svd


# Function to calculate reconstruction error
def reconstruction_error(U, Sigma, Vt, data, rank):
    """Compute reconstruction error for a given rank."""
    # Truncate to the desired rank
    U_r = U[:, :rank]
    Sigma_r = np.diag(Sigma[:rank])
    Vt_r = Vt[:rank, :]

    # Reconstruct the data
    data_reconstructed = U_r @ Sigma_r @ Vt_r

    # Compute the error
    error = np.linalg.norm(data - data_reconstructed, "fro")
    return error


# Optimization routine to find the best rank
def optimize_rank(data, max_rank=None, tolerance=1e-3):
    """
    Optimize the rank of the SVD by minimizing reconstruction error.

    Parameters:
        data (numpy.ndarray): The data matrix.
        max_rank (int): Maximum rank to consider. Default is None (full rank).
        tolerance (float): Desired reconstruction error tolerance.

    Returns:
        int: Optimal rank.
        list: Errors for each rank.
    """
    if max_rank is None:
        max_rank = min(data.shape)

    # Perform SVD on the data
    U, Sigma, Vt = svd(data, full_matrices=False)

    # Evaluate reconstruction error for each rank
    errors = []
    for rank in range(1, max_rank + 1):
        error = reconstruction_error(U, Sigma, Vt, data, rank)
        errors.append(error)
        if error <= tolerance:
            return rank, errors

    # If no rank satisfies the tolerance, return the max rank
    return max_rank, errors


# Example usage
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(0)
    data = np.random.rand(14, 49)  # Example data matrix

    # Find the optimal rank
    optimal_rank, errors = optimize_rank(data, tolerance=1e-6)

    print(f"Optimal rank: {optimal_rank}")

    # Plot the errors for visualization (optional)
    try:
        import matplotlib.pyplot as plt

        plt.plot(range(1, len(errors) + 1), errors, marker="o")
        plt.xlabel("Rank")
        plt.ylabel("Reconstruction Error")
        plt.title("Optimal Rank Selection")
        plt.grid(True)
        plt.show()
    except ImportError:
        print("matplotlib is not installed. Skipping plot.")
