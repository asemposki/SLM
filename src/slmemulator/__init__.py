# __init__.py for automatic discovery
import os
import sys

# Import everything from the listed sub-modules:
from .recombination import * # This will expose every function/class from recombination.py
from .SLM import *
from .TOV_class import *
from .plotData import *
from .config import * # This will expose get_paths (and anything else)
from .cleanData import * # This will expose clean_directory (and anything else)
from .pSLM import *
from .EOS_Codes import * # This will expose MSEOS (and anything else)

# This line is not needed if you use from .EOS_Codes import *
# from .EOS_Codes.Quarkyonia import generate_quarkyonia_eos

# [Rest of your config and path code...]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
__version__ = "0.1.0"