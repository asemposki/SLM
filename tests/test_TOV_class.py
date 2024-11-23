###########################################
# Unit tests of TOV_Class.py module
# Author: Alexandra C. Semposki
# Last edited: 14 November 2024
###########################################

import os
import numpy as np
import time

import sys
sys.path.append('../src')

from TOV_class import TOVSolver

from scipy.interpolate import interp1d

# test the TOVSolver class in 3 stages
filePath = os.getcwd().strip('src/')
eosName = "sorted_Sly4.dat"
fileName = filePath + "/EOS_Data/" + eosName

# 1: TOV solver for r, p, m, and Lambda
tov = TOVSolver(fileName, tidal=True)
total_radius, total_pres_central, total_mass = \
    tov.tov_routine(verbose=False, write_to_file=False)

# 2 Radius of 1.4 solar mass NS
rad_14 = tov.canonical_NS_radius()

# 3 Central density for max mass star
cdens = tov.central_dens()

###########################################################
# testing suite
###########################################################

# test if the SLy4 file exists
def test_file():
    file_path = '../EOS_Data/sorted_Sly4.dat'
    assert file_path is not None, "file does not exist"

# tests of the SLy4 results
def test_init():
    assert tov.eos_data is not None, "file not being parsed"
    assert tov.eps_array is not None, "file not read correctly"
    assert tov.pres_array is not None, "file not read correctly"
    assert tov.nB_array is not None, "file not read correctly"

def test_routine():
    assert tov.total_radius is not None, "solver not working"
    assert tov.total_pres_central is not None, "solver not working"
    assert tov.total_mass is not None, "solver not working"

    assert (tov.maximum_mass - 2.067) < 0.01, \
        "maximum mass more than 0.01 M_sol different"
    assert (tov.corr_radius - 10.032) < 0.1, \
        "radius more than 0.1 km different"
    assert (tov.corr_pres - 867.68) < 50.0, \
        "pressure more than 50 MeV/fm^3 different"
    
    assert np.isnan(tov.tidal_deformability).all() == False,\
        "tidal yields NaNs"
    
def test_14_star():
    assert (rad_14 - 11.788) < 0.1, \
        "Radius of 1.4 solar mass star differs more than 0.1 km"

def test_central_dens():
    assert (cdens - 1.18) < 0.1, \
        "Central density differs more than 0.1 fm^-3"