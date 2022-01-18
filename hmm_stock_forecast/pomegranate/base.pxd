# base.pxd
# Contact: Jacob Schreiber (jmschreiber91@gmail.com)

cdef class Model(object):
	cdef public str name
	cdef public int d
	cdef public bint frozen
	cdef public str model
	cdef public list states, edges
	cdef public object graph
	cdef int n_edges, n_states

	cdef void _log_probability(self, double* symbol, double* log_probability, int n) nogil
	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil


cdef class State(object):
	cdef public object distribution
	cdef public str name
	cdef public double weight
