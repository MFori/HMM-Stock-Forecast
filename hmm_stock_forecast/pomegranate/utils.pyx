# utils.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.math cimport log as clog
from libc.math cimport log2 as clog2
from libc.math cimport exp as cexp
from libc.math cimport floor
from libc.math cimport fabs
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport isnan

from scipy.linalg.cython_blas cimport dgemm


cimport cython
import numpy
cimport numpy

import numbers

import heapq

numpy.import_array()

cdef extern from "numpy/ndarraytypes.h":
	void PyArray_ENABLEFLAGS(numpy.ndarray X, int flags)

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463
DEF GAMMA = 0.577215664901532860606512090
DEF HALF_LOG2_PI = 0.91893853320467274178032973640562

cdef python_log_probability(model, double* X, double* log_probability, int n):
	cdef int i
	cdef numpy.npy_intp dim = n * model.d
	cdef numpy.ndarray X_ndarray

	X_ndarray = numpy.PyArray_SimpleNewFromData(1, &dim, numpy.NPY_FLOAT64, X)
	X_ndarray = X_ndarray.reshape(n, model.d)

	logp = model.log_probability(X_ndarray)
	
	if n == 1:
		log_probability[0] = logp
	else:
		for i in range(n):
			log_probability[i] = logp[i]

cdef python_summarize(model, double* X, double* weights, int n):
	cdef int i
	cdef numpy.npy_intp dim = n * model.d
	cdef numpy.npy_intp n_elements = n
	cdef numpy.ndarray X_ndarray
	cdef numpy.ndarray w_ndarray

	X_ndarray = numpy.PyArray_SimpleNewFromData(1, &dim, numpy.NPY_FLOAT64, X)
	if model.d > 1:
		X_ndarray = X_ndarray.reshape(n, model.d)

	w_ndarray = numpy.PyArray_SimpleNewFromData(1, &n_elements, 
		numpy.NPY_FLOAT64, weights)
	
	model.summarize(X_ndarray, w_ndarray)

# Useful speed optimized functions
cdef double _log(double x) nogil:
	'''
	A wrapper for the c log function, by returning negative infinity if the
	input is 0.
	'''
	return clog(x) if x > 0 else NEGINF

cdef double pair_lse(double x, double y) nogil:
	'''
	Perform log-sum-exp on a pair of numbers in log space..  This is calculated
	as z = log( e**x + e**y ). However, this causes underflow sometimes
	when x or y are too negative. A simplification of this is thus
	z = x + log( e**(y-x) + 1 ), where x is the greater number. If either of
	the inputs are infinity, return infinity, and if either of the inputs
	are negative infinity, then simply return the other input.
	'''

	if x == INF or y == INF:
		return INF
	if x == NEGINF:
		return y
	if y == NEGINF:
		return x
	if x > y:
		return x + clog(cexp(y-x) + 1)
	return y + clog(cexp(x-y) + 1)

def weight_set(items, weights):
	"""Converts both items and weights to appropriate numpy arrays.

	Convert the items into a numpy array with 64-bit floats, and the weight
	array to the same. If no weights are passed in, then return a numpy array
	with uniform weights.
	"""

	items = numpy.array(items, dtype=numpy.float64)
	if weights is None: # Weight everything 1 if no weights specified
		weights = numpy.ones(items.shape[0], dtype=numpy.float64)
	else: # Force whatever we have to be a Numpy array
		weights = numpy.asarray(weights, dtype=numpy.float64)

	return items, weights

def _check_nan(X):
	"""Checks to see if a value is nan, either as a float or a string."""
	if isinstance(X, (str, unicode, numpy.string_)):
		return X == 'nan'
	if isinstance(X, (float, numpy.float32, numpy.float64)):
		return isnan(X)
	return X is None
