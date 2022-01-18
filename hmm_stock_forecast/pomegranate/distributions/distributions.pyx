#!python
#cython: boundscheck=False
#cython: cdivision=True
# distributions.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>


import numpy
import sys

from ..utils import weight_set

cdef class Distribution(Model):
	"""A probability distribution.

	Represents a probability distribution over the defined support. This is
	the base class which must be subclassed to specific probability
	distributions. All distributions have the below methods exposed.

	Parameters
	----------
	Varies on distribution.

	Attributes
	----------
	name : str
		The name of the type of distributioon.

	summaries : list
		Sufficient statistics to store the update.

	frozen : bool
		Whether or not the distribution will be updated during training.

	d : int
		The dimensionality of the data. Univariate distributions are all
		1, while multivariate distributions are > 1.
	"""

	def __cinit__(self):
		self.name = "Distribution"
		self.frozen = False
		self.summaries = []
		self.d = 1

	def marginal(self, *args, **kwargs):
		"""Return the marginal of the distribution.

		Parameters
		----------
		*args : optional
			Arguments to pass in to specific distributions

		**kwargs : optional
			Keyword arguments to pass in to specific distributions

		Returns
		-------
		distribution : Distribution
			The marginal distribution. If this is a multivariate distribution
			then this method is filled in. Otherwise returns self.
		"""

		return self

	def copy(self):
		"""Return a deep copy of this distribution object.

		This object will not be tied to any other distribution or connected
		in any form.

		Paramters
		---------
		None

		Returns
		-------
		distribution : Distribution
			A copy of the distribution with the same parameters.
		"""

		return self.__class__(*self.parameters)

	def log_probability(self, X):
		"""Return the log probability of the given X under this distribution.

		Parameters
		----------
		X : double
			The X to calculate the log probability of (overridden for
			DiscreteDistributions)

		Returns
		-------
		logp : double
			The log probability of that point under the distribution.
		"""

		cdef int i, n
		cdef double logp
		cdef numpy.ndarray logp_array
		cdef numpy.ndarray X_ndarray
		cdef double* X_ptr
		cdef double* logp_ptr

		n = 1 if isinstance(X, (int, float)) else len(X)

		logp_array = numpy.empty(n, dtype='float64')
		logp_ptr = <double*> logp_array.data

		X_ndarray = numpy.asarray(X, dtype='float64')
		X_ptr = <double*> X_ndarray.data

		self._log_probability(X_ptr, logp_ptr, n)

		if n == 1:
			return logp_array[0]
		else:
			return logp_array

	def fit(self, items, weights=None, inertia=0.0, column_idx=0):
		"""
		Set the parameters of this Distribution to maximize the likelihood of
		the given sample. Items holds some sort of sequence. If weights is
		specified, it holds a sequence of value to weight each item by.
		"""

		if self.frozen:
			return

		self.summarize(items, weights, column_idx)
		self.from_summaries(inertia)

	def summarize(self, items, weights=None, column_idx=0):
		"""
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		"""

		items, weights = weight_set(items, weights)
		if weights.sum() <= 0:
			return

		cdef double* items_p = <double*> (<numpy.ndarray> items).data
		cdef double* weights_p = <double*> (<numpy.ndarray> weights).data
		cdef int n = items.shape[0]
		cdef int d = 1
		cdef int column_id = <int> column_idx

		if items.ndim == 2:
			d = items.shape[1]

		with nogil:
			self._summarize(items_p, weights_p, n, column_id, d)

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		pass

	def from_summaries(self, inertia=0.0):
		"""Fit the distribution to the stored sufficient statistics.
		"""
		pass

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object.
		"""
		pass

	@classmethod
	def from_samples(cls, items, weights=None, **kwargs):
		"""Fit a distribution to some data without pre-specifying it."""

		distribution = cls.blank()
		distribution.fit(items, weights, **kwargs)
		return distribution