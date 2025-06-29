import os
import sys
from .recombination import *
from .SLM import *
from .TOV_class import TOV
from .plotData import *
from .config import get_paths
from .cleanData import *
from .pSLM import *
from .EOS_Codes import Quarkyonia, MSEOS
from .EOS_Codes.Quarkyonia import generate_quarkyonia_eos

# Add base directory (project root) to sys path
# If __file__ is /project_root/src/slmemulator/__init__.py,
# then this correctly calculates /project_root/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Versioning (good practice)
__version__ = "0.1.0"
