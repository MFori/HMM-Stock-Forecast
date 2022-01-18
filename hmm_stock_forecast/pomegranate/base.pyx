# base.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from .utils cimport *

import numpy
import uuid


# Define some useful constants
DEF NEGINF = float("-inf")

cdef class Model(object):
	"""The abstract building block for all distributions."""

	def __cinit__(self):
		self.name = "Model"

	def __init__(self, name=None):
		"""
        Make a new graphical model. Name is an optional string used to name
        the model when output. Name may not contain spaces or newlines.
        """

		# Save the name or make up a name.
		self.name = name or str(id(self))
		self.states = []
		self.edges = []

	def get_params(self, *args, **kwargs):
		return self.__getstate__()

	def set_params(self, state):
		self.__setstate__(state)

	cdef void _log_probability(self, double* symbol, double* log_probability,
		int n) nogil:
		pass

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		pass


cdef class State(object):
	"""
	Represents a state in an HMM. Holds emission distribution, but not
	transition distribution, because that's stored in the graph edges.
	"""

	def __init__( self, distribution, name=None, weight=None ):
		"""
		Make a new State emitting from the given distribution. If distribution
		is None, this state does not emit anything. A name, if specified, will
		be the state's name when presented in output. Name may not contain
		spaces or newlines, and must be unique within a model.
		"""

		self.distribution = distribution
		self.name = name or str(uuid.uuid4())
		self.weight = weight or 1.

	def __reduce__(self):
		return self.__class__, (self.distribution, self.name, self.weight)

	def tie( self, state ):
		"""
		Tie this state to another state by just setting the distribution of the
		other state to point to this states distribution.
		"""
		state.distribution = self.distribution

	def is_silent(self):
		"""
		Return True if this state is silent (distribution is None) and False
		otherwise.
		"""
		return self.distribution is None


# Create a convenient alias
Node = State
