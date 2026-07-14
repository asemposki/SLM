"""Recombination thinning for the Banach GRIM algorithm (arXiv:2205.07495).

The implementation lives in banach_grim.py (so that module stays
self-contained); this module re-exports it alongside the Gaussian RBF helper.
"""

import numpy as np

from .banach_grim import recombination_thinning

__all__ = ["gaussian_rbf", "recombination_thinning"]


def gaussian_rbf(r, epsilon):
    """Gaussian Radial Basis Function."""
    return np.exp(-((epsilon * r) ** 2))
