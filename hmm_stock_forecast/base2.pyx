# base.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")


cdef class Hovno(object):
	"""The abstract building block for all distributions."""

	def __init__(self):
		print("hovno initted")

	def to_dict(self):
		"""Serialize this object to a dictionary of parameters."""
		raise NotImplementedError

	def copy(self):
		"""Return a deep copy of this distribution object.

		This object will not be tied to any other distribution or connected
		in any form.

		Parameters
		----------
		None

		Returns
		-------
		distribution : Distribution
			A copy of the distribution with the same parameters.
		"""

		return self.__class__.from_json( self.to_json() )

	def freeze(self):
		"""Freeze the distribution, preventing updates from occurring."""
		self.frozen = True

	def thaw(self):
		"""Thaw the distribution, re-allowing updates to occur."""
		self.frozen = False

	def copy(self):
		"""Return a deep copy of this distribution object.

		This object will not be tied to any other distribution or connected
		in any form.

		Parameters
		----------
		None

		Returns
		-------
		distribution : Distribution
			A copy of the distribution with the same parameters.
		"""

		return self.__class__(*self.parameters)

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

	def score(self, X, y):
		"""Return the accuracy of the model on a data set.

		Parameters
		----------
		X : numpy.ndarray, shape=(n, d)
			The values of the data set

		y : numpy.ndarray, shape=(n,)
			The labels of each value
		"""

		return (self.predict(X) == y).mean()


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