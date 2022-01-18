# __init__.py: pomegranate
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>


"""
For detailed documentation and examples, see the README.
"""

import os

from .base import *
#from .parallel import *

from .distributions import *
from .kmeans import Kmeans
from .hmm import HiddenMarkovModel

__version__ = '1.0'

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
