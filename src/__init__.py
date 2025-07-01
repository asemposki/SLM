import os
import sys
# from .recombination import *

# Add base directory to sys path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Define other directory paths as needed
SRC_DIR = os.path.join(BASE_DIR, "src")
EOS_CODES_DIR = os.path.join(BASE_DIR, "EOS_Codes")
EOS_DATA_DIR = os.path.join(BASE_DIR, "EOS_Data")
EOS_FILES_DIR = os.path.join(BASE_DIR, "EOS_files")
RESULTS_PATH = os.path.join(BASE_DIR, "Results")
TOV_PATH = os.path.join(BASE_DIR, "TOV_data")
PLOTS_PATH = os.path.join(BASE_DIR, "Plots")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
TRAIN_PATH = os.path.join(BASE_DIR, "trainData")
TESTS_DIR = os.path.join(BASE_DIR, "tests")
TEST_DATA_PATH = os.path.join(BASE_DIR, "testData")
TUTORIALS_DIR = os.path.join(BASE_DIR, "Tutorials")

# Quarkyonia and MSEOS paths
QEOS_PATH = os.path.join(EOS_FILES_DIR, "QEOS")
MSEOS_PATH = os.path.join(EOS_FILES_DIR, "MSEOS")

# Quarkyonia and MS EOS TOV results paths
QEOS_TOV_PATH = os.path.join(TOV_PATH, "QEOS")
MSEOS_TOV_PATH = os.path.join(TOV_PATH, "MSEOS")

# SLM results path
SLM_RES_MSEOS = os.path.join(RESULTS_PATH, "MSEOS")
SLM_RES_QEOS = os.path.join(RESULTS_PATH, "QEOS")
