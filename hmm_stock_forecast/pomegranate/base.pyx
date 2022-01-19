DEF NEGINF = float("-inf")

cdef class Model(object):
	"""The abstract building block for all distributions."""

	def __init__(self):
		"""
        Make a new graphical model. Name is an optional string used to name
        the model when output. Name may not contain spaces or newlines.
        """

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
