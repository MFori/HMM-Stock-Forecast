# NormalDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

cimport numpy

ctypedef numpy.npy_float64 DOUBLE_t
ctypedef numpy.npy_intp SIZE_t

cdef class NormalDistribution(object):
	cdef public str name
	cdef public int d
	cdef public bint frozen
	cdef public str model
	cdef public list summaries
	cdef double mu, sigma, two_sigma_squared, log_sigma_sqrt_2_pi
	cdef object min_std

	#cdef void _log_probability(self, double* symbol, double* log_probability, int n) nogil
	#cdef double _summarize(self, double* items, double* weights, int n,
	#					   int column_idx, int d) nogil

