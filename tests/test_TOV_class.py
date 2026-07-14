###########################################
# Unit tests of the TOV_class.py module
# Author: Alexandra C. Semposki
###########################################

import numpy as np
import pytest

from slmemulator import TOV, get_paths


@pytest.fixture(scope="module")
def tov():
    """Solve the TOV equations once for the packaged SLy4 EOS."""
    eos_file = get_paths()["package_eos_data_dir"] / "sorted_Sly4.dat"
    assert eos_file.is_file(), "packaged SLy4 EOS file is missing"

    solver = TOV(eos_file, tidal=True)
    solver.tov_routine(verbose=False, write_to_file=False)
    return solver


def test_init(tov):
    assert tov.eps_array is not None, "file not read correctly"
    assert tov.pres_array is not None, "file not read correctly"
    assert tov.nB_array is not None, "file not read correctly"


def test_routine(tov):
    assert tov.total_radius is not None, "solver not working"
    assert tov.total_pres_central is not None, "solver not working"
    assert tov.total_mass is not None, "solver not working"

    assert abs(tov.maximum_mass - 2.067) < 0.01, (
        "maximum mass more than 0.01 M_sol different"
    )
    assert abs(tov.corr_radius - 10.09) < 0.1, "radius more than 0.1 km different"
    assert abs(tov.corr_pres - 820.97) < 50.0, (
        "pressure more than 50 MeV/fm^3 different"
    )

    assert np.isfinite(tov.tidal_deformability).all(), "tidal yields NaNs"
    assert np.isfinite(tov.k2).all(), "k2 yields NaNs"


def test_14_star(tov):
    rad_14 = tov.canonical_NS_radius()
    assert abs(rad_14 - 11.79) < 0.1, (
        "Radius of 1.4 solar mass star differs more than 0.1 km"
    )


def test_central_dens(tov):
    cdens = tov.central_dens()
    assert abs(cdens - 1.19) < 0.1, "Central density differs more than 0.1 fm^-3"


def test_non_tidal_routine():
    """The non-tidal path must run and return three arrays (regression:
    it used to crash on np.column_stack(None))."""
    eos_file = get_paths()["package_eos_data_dir"] / "sorted_Sly4.dat"
    solver = TOV(eos_file, tidal=False, sol_pts=500)
    radius, pres_central, mass = solver.tov_routine()
    assert radius.shape == pres_central.shape == mass.shape
    assert np.max(mass) > 1.0, "maximum mass suspiciously small"
