# distributions.pxd
# Contact: Jacob Schreiber (jmschreiber91@gmail.com)

cimport numpy

ctypedef numpy.npy_float64 DOUBLE_t
ctypedef numpy.npy_intp SIZE_t

cdef class Distribution(object):
	cdef public str name
	cdef public int d
	cdef public bint frozen
	cdef public str model
	cdef public list summaries

	cdef void _log_probability(self, double* symbol, double* log_probability, int n) nogil
	cdef double _vl_log_probability(self, double* symbol, int n) nogil
	cdef double _summarize(self, double* items, double* weights, int n,
						   int column_idx, int d) nogil

