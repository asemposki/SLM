# src/SLM/__init__.py

# Expose key classes/functions directly under the 'SLM' namespace
from .SLM import *
from .p2SLM import *

# Optionally, expose common functions from recombination.py directly
from .recombination import recombine_data

# Versioning (good practice)
__version__ = "0.1.0"
