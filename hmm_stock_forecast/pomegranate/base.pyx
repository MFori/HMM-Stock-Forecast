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

	def sample(self, n=None):
		"""Return a random item sampled from this distribution.

		Parameters
		----------
		n : int or None, optional
			The number of samples to return. Default is None, which is to
			generate a single sample.

		Returns
		-------
		sample : double or object
			Returns a sample from the distribution of a type in the support
			of the distribution.
		"""

		raise NotImplementedError

	def probability(self, symbol):
		"""Return the probability of the given symbol under this distribution.

		Parameters
		----------
		symbol : object
			The symbol to calculate the probability of

		Returns
		-------
		probability : double
			The probability of that point under the distribution.
		"""

		return numpy.exp(self.log_probability(symbol))

	def log_probability(self, symbol):
		"""Return the log probability of the given symbol under this
		distribution.

		Parameters
		----------
		symbol : double
			The symbol to calculate the log probability of (overridden for
			DiscreteDistributions)

		Returns
		-------
		logp : double
			The log probability of that point under the distribution.
		"""

		raise NotImplementedError

	def sample(self, n=None):
		"""Return a random item sampled from this distribution.

		Parameters
		----------
		n : int or None, optional
			The number of samples to return. Default is None, which is to
			generate a single sample.

		Returns
		-------
		sample : double or object
			Returns a sample from the distribution of a type in the support
			of the distribution.
		"""

		raise NotImplementedError

	def fit(self, items, weights=None, inertia=0.0):
		"""Fit the distribution to new data using MLE estimates.

		Parameters
		----------
		items : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on. For univariate distributions an array
			is used, while for multivariate distributions a 2d matrix is used.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be
			old_param * inertia + new_param * (1-inertia), so an inertia of 0
			means ignore the old parameters, whereas an inertia of 1 means
			ignore the new parameters. Default is 0.0.

		Returns
		-------
		None
		"""

		raise NotImplementedError

	def summarize(self, items, weights=None):
		"""Summarize a batch of data into sufficient statistics for a later
		update.

		Parameters
		----------
		items : array-like, shape (n_samples, n_dimensions)
			This is the data to train on. Each row is a sample, and each column
			is a dimension to train on. For univariate distributions an array
			is used, while for multivariate distributions a 2d matrix is used.

		weights : array-like, shape (n_samples,), optional
			The initial weights of each sample in the matrix. If nothing is
			passed in then each sample is assumed to be the same weight.
			Default is None.

		Returns
		-------
		None
		"""

		return NotImplementedError

	def from_summaries(self, inertia=0.0):
		"""Fit the distribution to the stored sufficient statistics.

		Parameters
		----------
		inertia : double, optional
			The weight of the previous parameters of the model. The new
			parameters will roughly be
			old_param * inertia + new_param * (1-inertia), so an inertia of 0
			means ignore the old parameters, whereas an inertia of 1 means
			ignore the new parameters. Default is 0.0.

		Returns
		-------
		None
		"""

		return NotImplementedError

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""
		return NotImplementedError

	cdef void _log_probability(self, double* symbol, double* log_probability,
		int n) nogil:
		pass

	cdef double _vl_log_probability(self, double* symbol, int n) nogil:
		return NEGINF

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		pass

	def add_edge(self, a, b):
		"""
		Add a transition from state a to state b which indicates that B is
		dependent on A in ways specified by the distribution.
		"""

		# Add the transition
		self.edges.append((a, b))
		self.n_edges += 1


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

		# Save the distribution
		self.distribution = distribution

		# Save the name
		self.name = name or str(uuid.uuid4())

		# Save the weight, or default to the unit weight
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

	def tied_copy(self):
		"""
		Return a copy of this state where the distribution is tied to the
		distribution of this state.
		"""
		return State( distribution=self.distribution, name=self.name+'-tied' )

	def copy( self ):
		"""Return a hard copy of this state."""
		return State( distribution=self.distribution.copy(), name=self.name )


# Create a convenient alias
Node = State
