from libc.math cimport exp as cexp
from operator import attrgetter
import networkx
import time

from . import NormalDistribution, State
from .base cimport Model

from sklearn.cluster import KMeans

from .utils cimport _log
from .utils cimport pair_lse
from .utils cimport python_log_probability
from .utils cimport python_summarize

from .io import SequenceGenerator

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.string cimport memset

import numpy
cimport numpy

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463

cdef class HiddenMarkovModel(Model):
	"""A Hidden Markov Model

	A Hidden Markov Model (HMM) is a directed graphical model where nodes are
	hidden states which contain an observed emission distribution and edges
	contain the probability of transitioning from one hidden state to another.
	HMMs allow you to tag each observation in a variable length sequence with
	the most likely hidden state according to the model.

	Parameters
	----------

	start : State, optional
		An optional state to force the model to start in. Default is None.

	end : State, optional
		An optional state to force the model to end in. Default is None.

	Attributes
	----------
	start : State
		A state object corresponding to the initial start of the model

	end : State
		A state object corresponding to the forced end of the model

	start_index : int
		The index of the start object in the state list

	silent_start : int
		The index of the beginning of the silent states in the state list

	states : list
		The list of all states in the model, with silent states at the end

	Examples
	--------
	"""

	cdef public object start, end
	cdef public int start_index
	cdef public int silent_start
	cdef double* in_transition_pseudocounts
	cdef double* out_transition_pseudocounts
	cdef double [:] state_weights
	cdef int summaries
	cdef int* tied_state_count
	cdef int* tied
	cdef int* tied_edge_group_size
	cdef int* tied_edges_starts
	cdef int* tied_edges_ends
	cdef double* in_transition_log_probabilities
	cdef double* out_transition_log_probabilities
	cdef double* expected_transitions
	cdef int* in_edge_count
	cdef int* in_transitions
	cdef int* out_edge_count
	cdef int* out_transitions
	cdef int n_tied_edge_groups
	cdef numpy.ndarray distributions
	cdef void** distributions_ptr

	def __init__(self, start=None, end=None):
		# This holds a directed graph between states. Nodes in that graph are
		# State objects, so they're guaranteed never to conflict when composing
		# two distinct models
		self.graph = networkx.DiGraph()

		# Save the start and end or mae one up
		self.start = start or State(None, name="start")
		self.end = end or State(None, name="end")

		# Put start and end in the graph
		self.graph.add_node(self.start)
		self.graph.add_node(self.end)

	def __dealloc__(self):
		self.free_bake_buffers()

	def free_bake_buffers(self):
		free(self.in_transition_pseudocounts)
		free(self.out_transition_pseudocounts)
		free(self.tied_state_count)
		free(self.tied)
		free(self.tied_edge_group_size)
		free(self.tied_edges_starts)
		free(self.tied_edges_ends)
		free(self.in_transition_log_probabilities)
		free(self.out_transition_log_probabilities)
		free(self.expected_transitions)
		free(self.in_edge_count)
		free(self.in_transitions)
		free(self.out_edge_count)
		free(self.out_transitions)


	def add_state(self, state):
		self.graph.add_node(state)

	def add_transition(self, a, b, probability, pseudocount=None, group=None):
		"""Add a transition from state a to state b.

		Add a transition from state a to state b with the given (non-log)
		probability. Both states must be in the HMM already. self.start and
		self.end are valid arguments here. Probabilities will be normalized
		such that every node has edges summing to 1. leaving that node, but
		only when the model is baked. Psueodocounts are allowed as a way of
		using edge-specific pseudocounts for training.

		By specifying a group as a string, you can tie edges together by giving
		them the same group. This means that a transition across one edge in the
		group counts as a transition across all edges in terms of training.

		Parameters
		----------
		a : State
			The state that the edge originates from

		b : State
			The state that the edge goes to

		probability : double
			The probability of transitioning from state a to state b in [0, 1]

		pseudocount : double, optional
			The pseudocount to use for this specific edge if using edge
			pseudocounts for training. Defaults to the probability. Default
			is None.

		group : str, optional
			The name of the group of edges to tie together during training. If
			groups are used, then a transition across any one edge counts as a
			transition across all edges. Default is None.

		Returns
		-------
		None
		"""

		pseudocount = pseudocount or probability
		self.graph.add_edge(a, b, probability=_log(probability),
			pseudocount=pseudocount, group=group)

	def bake(self):
		"""Finalize the topology of the model.

		Finalize the topology of the model and assign a numerical index to
		every state. This method must be called before any of the probability-
		calculating methods.

		This fills in self.states (a list of all states in order) and
		self.transition_log_probabilities (log probabilities for transitions),
		as well as self.start_index and self.end_index, and self.silent_start
		(the index of the first silent state).
		"""

		if len(self.graph.nodes()) == 0:
			raise ValueError("Must add states to the model before baking it.")

		cdef int i

		self.free_bake_buffers()

		in_edge_count = numpy.zeros(len(self.graph.nodes()),
			dtype=numpy.int32)
		out_edge_count = numpy.zeros(len(self.graph.nodes()),
			dtype=numpy.int32)

		states = self.graph.nodes()
		n, m = len(states), len(self.graph.edges())

		self.n_edges = m
		self.n_states = n

		silent_states, normal_states = [], []

		for state in states:
			if state.is_silent():
				silent_states.append(state)
			else:
				normal_states.append(state)

		normal_states = list(sorted(normal_states, key=attrgetter('name')))
		silent_states = list(sorted(silent_states, key=attrgetter('name')))
		silent_order = {state: i for i, state in enumerate(reversed(silent_states))}

		# We need the silent states to be in topological sort order: any
		# transition between silent states must be from a lower-numbered state
		# to a higher-numbered state. Since we ban loops of silent states, we
		# can get away with this.

		# Get the subgraph of all silent states
		silent_subgraph = self.graph.subgraph(silent_states)

		# Get the sorted silent states. Isn't it convenient how NetworkX has
		# exactly the algorithm we need?
		silent_states_sorted = list(networkx.lexicographical_topological_sort(
			silent_subgraph, silent_order.__getitem__))

		# What's the index of the first silent state?
		self.silent_start = len(normal_states)

		# Save the master state ordering. Silent states are last and in
		# topological order, so when calculationg forward algorithm
		# probabilities we can just go down the list of states.
		self.states = normal_states + silent_states_sorted

		# We need a good way to get transition probabilities by state index that
		# isn't N^2 to build or store. So we will need a reverse of the above
		# mapping. It's awkward but asymptotically fine.
		indices = { self.states[i]: i for i in range(n) }

		# Create a sparse representation of the tied states in the model. This
		# is done in the same way of the transition, by having a vector of
		# counts, and a vector of the IDs that the state is tied to.
		self.tied_state_count = <int*> calloc(self.silent_start+1, sizeof(int))

		for i in range(self.silent_start):
			for j in range(self.silent_start):
				if i == j:
					continue
				if self.states[i].distribution is self.states[j].distribution:
					self.tied_state_count[i+1] += 1

		for i in range(1, self.silent_start+1):
			self.tied_state_count[i] += self.tied_state_count[i-1]

		self.tied = <int*> malloc(self.tied_state_count[self.silent_start]*sizeof(int))
		for i in range(self.tied_state_count[self.silent_start]):
			self.tied[i] = -1

		for i in range(self.silent_start):
			for j in range(self.silent_start):
				if i == j:
					continue

				if self.states[i].distribution is self.states[j].distribution:
					# Begin at the first index which belongs to state i...
					start = self.tied_state_count[i]

					# Find the first non -1 entry in order to put our index.
					while self.tied[start] != -1:
						start += 1

					# Now that we've found a non -1 entry, put the index of the
					# state which this state is tied to in!
					self.tied[start] = j

		# Unpack the state weights
		self.state_weights = numpy.empty(self.silent_start)
		for i in range(self.silent_start):
			self.state_weights[i] = _log(self.states[i].weight)

		# This holds numpy array indexed [a, b] to transition log probabilities
		# from a to b, where a and b are state indices. It starts out saying all
		# transitions are impossible.
		self.in_transitions = <int*> malloc(m*sizeof(int))
		self.in_edge_count = <int*> calloc(n+1, sizeof(int))
		self.in_transition_pseudocounts = <double*> calloc(m,
			sizeof(double))
		self.in_transition_log_probabilities = <double*> calloc(m,
			sizeof(double))

		self.out_transitions = <int*> malloc(m*sizeof(int))
		self.out_edge_count = <int*> calloc(n+1, sizeof(int))
		self.out_transition_pseudocounts = <double*> calloc(m,
			sizeof(double))
		self.out_transition_log_probabilities = <double*> calloc(m,
			sizeof(double))

		self.expected_transitions =  <double*> calloc(self.n_edges, sizeof(double))

		memset(self.in_transitions, -1, m*sizeof(int))
		memset(self.out_transitions, -1, m*sizeof(int))

		# Now we need to find a way of storing in-edges for a state in a manner
		# that can be called in the cythonized methods below. This is basically
		# an inversion of the graph. We will do this by having two lists, one
		# list size number of nodes + 1, and one list size number of edges.
		# The node size list will store the beginning and end values in the
		# edge list that point to that node. The edge list will be ordered in
		# such a manner that all edges pointing to the same node are grouped
		# together. This will allow us to run the algorithms in time
		# nodes*edges instead of nodes*nodes.
		for a, b in self.graph.edges():
			# Increment the total number of edges going to node b.
			self.in_edge_count[indices[b]+1] += 1
			# Increment the total number of edges leaving node a.
			self.out_edge_count[indices[a]+1] += 1

		# Take the cumulative sum so that we can associate array indices with
		# in or out transitions
		for i in range(1, n+1):
			self.in_edge_count[i] += self.in_edge_count[i-1]
			self.out_edge_count[i] += self.out_edge_count[i-1]

		# We need to store the edge groups as name : set pairs.
		edge_groups = {}

		# Now we go through the edges again in order to both fill in the
		# transition probability matrix, and also to store the indices sorted
		# by the end-node.
		for a, b, data in self.graph.edges(data=True):
			# Put the edge in the dict. Its weight is log-probability
			start = self.in_edge_count[indices[b]]

			# Start at the beginning of the section marked off for node b.
			# If another node is already there, keep walking down the list
			# until you find a -1 meaning a node hasn't been put there yet.
			while self.in_transitions[start] != -1:
				if start == self.in_edge_count[indices[b]+1]:
					break
				start += 1

			self.in_transition_log_probabilities[start] = <double>data['probability']
			self.in_transition_pseudocounts[start] = data['pseudocount']

			# Store transition info in an array where the in_edge_count shows
			# the mapping stuff.
			self.in_transitions[start] = <int>indices[a]

			# Now do the same for out edges
			start = self.out_edge_count[indices[a]]

			while self.out_transitions[start] != -1:
				if start == self.out_edge_count[indices[a]+1]:
					break
				start += 1

			self.out_transition_log_probabilities[start] = <double>data['probability']
			self.out_transition_pseudocounts[start] = data['pseudocount']
			self.out_transitions[start] = <int>indices[b]

			# If this edge belongs to a group, we need to add it to the
			# dictionary. We only care about forward representations of
			# the edges.
			group = data['group']
			if group != None:
				if group in edge_groups:
					edge_groups[group].append((indices[a], indices[b]))
				else:
					edge_groups[group] = [(indices[a], indices[b])]

		# We will organize the tied edges using three arrays. The first will be
		# the cumulative number of members in each group, to slice the later
		# arrays in the same manner as the transition arrays. The second will
		# be the index of the state the edge starts in. The third will be the
		# index of the state the edge ends in. This way, iterating across the
		# second and third lists in the slices indicated by the first list will
		# give all the edges in a group.
		total_grouped_edges = sum(map(len, edge_groups.values()))

		self.n_tied_edge_groups = len(edge_groups.keys())+1
		self.tied_edge_group_size = <int*> malloc((len(edge_groups.keys())+1)*
			sizeof(int))
		self.tied_edge_group_size[0] = 0

		self.tied_edges_starts = <int*> calloc(total_grouped_edges, sizeof(int))
		self.tied_edges_ends = <int*> calloc(total_grouped_edges, sizeof(int))

		# Iterate across all the grouped edges and bin them appropriately.
		for i, (name, edges) in enumerate(edge_groups.items()):
			# Store the cumulative number of edges so far, which requires
			# adding the current number of edges (m) to the previous
			# number of edges (n)
			n = self.tied_edge_group_size[i]
			self.tied_edge_group_size[i+1] = n + len(edges)

			for j, (start, end) in enumerate(edges):
				self.tied_edges_starts[n+j] = start
				self.tied_edges_ends[n+j] = end

		for state in self.states:
			if not state.is_silent():
				dist = state.distribution
				break

		self.d = dist.d

		if self.d > 1:
			for state in self.states[:self.silent_start]:
				d = state.distribution

		self.distributions = numpy.empty(self.silent_start, dtype='object')
		for i in range(self.silent_start):
			self.distributions[i] = self.states[i].distribution
			if self.d != self.distributions[i].d:
				raise ValueError("mis-matching inputs for states")

		self.distributions_ptr = <void**> self.distributions.data
		self.start_index = indices[self.start]

	def log_probability(self, sequence, check_input=True):
		"""Calculate the log probability of a single sequence.

		If a path is provided, calculate the log probability of that sequence
		given the path.

		Parameters
		----------
		sequence : array-like
			Return the array of observations in a single sequence of data

		check_input : bool, optional
			Check to make sure that all emissions fall under the support of
			the emission distributions. Default is True.

		Returns
		-------
		logp : double
			The log probability of the sequence
		"""

		if self.d == 0:
			raise ValueError("must bake model before computing probability")

		cdef numpy.ndarray sequence_ndarray
		cdef double* sequence_ptr
		cdef double log_probability
		cdef int n = len(sequence)

		sequence_ndarray = sequence

		sequence_ptr = <double*> sequence_ndarray.data

		with nogil:
			log_probability = self._vl_log_probability(sequence_ptr, n)

		return log_probability

	cdef double _vl_log_probability(self, double* sequence, int n) nogil:
		cdef double* f = self._forward(sequence, n, NULL)
		cdef double log_probability
		cdef int i, m = self.n_states

		log_probability = NEGINF
		for i in range(self.silent_start):
			log_probability = pair_lse(log_probability, f[n*m + i])

		free(f)
		return log_probability

	cdef double* _forward(self, double* sequence, int n, double* emissions) nogil:
		cdef int i, k, ki, l, li
		cdef int p = self.silent_start, m = self.n_states
		cdef int dim = self.d

		cdef void** distributions = <void**> self.distributions_ptr

		cdef double log_probability
		cdef int* in_edges = self.in_edge_count

		cdef double* e = NULL
		cdef double* f = <double*> calloc(m*(n+1), sizeof(double))

		# Either fill in a new emissions matrix, or use the one which has
		# been provided from a previous call.
		if emissions is NULL:
			e = <double*> malloc(n*self.silent_start*sizeof(double))
			for l in range(self.silent_start):
				for i in range(n):
					with gil:
						python_log_probability(self.distributions[l], sequence+i*dim, e+l*n+i, 1)

					e[l*n + i] += self.state_weights[l]
		else:
			e = emissions

		# We must start in the start state, having emitted 0 symbols
		for i in range(m):
			f[i] = NEGINF
		f[self.start_index] = 0.

		for l in range(self.silent_start, m):
			# Handle transitions between silent states before the first symbol
			# is emitted. No non-silent states have non-zero probability yet, so
			# we can ignore them.
			if l == self.start_index:
				# Start state log-probability is already right. Don't touch it.
				continue

			# This holds the log total transition probability in from
			# all current-step silent states that can have transitions into
			# this state.
			log_probability = NEGINF
			for k in range(in_edges[l], in_edges[l+1]):
				ki = self.in_transitions[k]
				if ki < self.silent_start or ki >= l:
					continue

				# For each current-step preceding silent state k
				log_probability = pair_lse(log_probability,
					f[ki] + self.in_transition_log_probabilities[k])

			# Update the table entry
			f[l] = log_probability

		for i in range(n):
			for l in range(self.silent_start):
				# Do the recurrence for non-silent states l
				# This holds the log total transition probability in from
				# all previous states

				log_probability = NEGINF
				for k in range(in_edges[l], in_edges[l+1]):
					ki = self.in_transitions[k]

					# For each previous state k
					log_probability = pair_lse(log_probability,
						f[i*m + ki] + self.in_transition_log_probabilities[k])

				# Now set the table entry for log probability of emitting
				# index+1 characters and ending in state l
				f[(i+1)*m + l] = log_probability + e[i + l*n]

			for l in range(self.silent_start, m):
				# Now do the first pass over the silent states
				# This holds the log total transition probability in from
				# all current-step non-silent states
				log_probability = NEGINF
				for k in range(in_edges[l], in_edges[l+1]):
					ki = self.in_transitions[k]
					if ki >= self.silent_start:
						continue

					# For each current-step non-silent state k
					log_probability = pair_lse(log_probability,
						f[(i+1)*m + ki] + self.in_transition_log_probabilities[k])

				# Set the table entry to the partial result.
				f[(i+1)*m + l] = log_probability

			for l in range(self.silent_start, m):
				# Now the second pass through silent states, where we account
				# for transitions between silent states.

				# This holds the log total transition probability in from
				# all current-step silent states that can have transitions into
				# this state.
				log_probability = NEGINF
				for k in range(in_edges[l], in_edges[l+1]):
					ki = self.in_transitions[k]
					if ki < self.silent_start or ki >= l:
						continue
					# For each current-step preceding silent state k
					log_probability = pair_lse(log_probability,
						f[(i+1)*m + ki] + self.in_transition_log_probabilities[k])

				# Add the previous partial result and update the table entry
				f[(i+1)*m + l] = pair_lse(f[(i+1)*m + l], log_probability)

		if emissions is NULL:
			free(e)
		return f

	cdef double* _backward(self, double* sequence, int n, double* emissions) nogil:
		cdef int i, ir, k, kr, l, li
		cdef int p = self.silent_start, m = self.n_states
		cdef int dim = self.d

		cdef void** distributions = <void**> self.distributions_ptr

		cdef double log_probability
		cdef int* out_edges = self.out_edge_count

		cdef double* e = NULL
		cdef double* b = <double*> calloc((n+1)*m, sizeof(double))

		# Either fill in a new emissions matrix, or use the one which has
		# been provided from a previous call.
		if emissions is NULL:
			e = <double*> malloc(n*self.silent_start*sizeof(double))
			for l in range(self.silent_start):
				for i in range(n):
					with gil:
						python_log_probability(self.distributions[l], sequence+i*dim, e+l*n+i, 1)

					e[l*n + i] += self.state_weights[l]
		else:
			e = emissions

		# We must end in the end state, having emitted len(sequence) symbols
		for i in range(self.silent_start):
			b[n*m + i] = 0.
		for i in range(self.silent_start, m):
			b[n*m + i] = NEGINF

		# Now that we're done with the base case, move on to the recurrence
		for ir in range(n):
			#if self.finite == 0 and ir == 0:
			#   continue
			# Cython ranges cannot go backwards properly, redo to handle
			# it properly
			i = n - ir - 1
			for kr in range(m-self.silent_start):
				k = m - kr - 1

				# Do the silent states' dependency on subsequent non-silent
				# states, iterating backwards to match the order we use later.

				# This holds the log total probability that we go to some
				# subsequent state that emits the right thing, and then continue
				# from there to finish the sequence.
				log_probability = NEGINF
				for l in range(out_edges[k], out_edges[k+1]):
					li = self.out_transitions[l]
					if li >= self.silent_start:
						continue

					# For each subsequent non-silent state l, take into account
					# transition and emission emission probability.
					log_probability = pair_lse(log_probability,
						b[(i+1)*m + li] + self.out_transition_log_probabilities[l] +
						e[i + li*n])

				# We can't go from a silent state here to a silent state on the
				# next symbol, so we're done finding the probability assuming we
				# transition straight to a non-silent state.
				b[i*m + k] = log_probability

			for kr in range(m-self.silent_start):
				k = m - kr - 1

				# Do the silent states' dependencies on each other.
				# Doing it in reverse order ensures that anything we can
				# possibly transition to is already done.

				# This holds the log total probability that we go to
				# current-step silent states and then continue from there to
				# finish the sequence.
				log_probability = NEGINF
				for l in range(out_edges[k], out_edges[k+1]):
					li = self.out_transitions[l]
					if li < k+1:
						continue

					# For each possible current-step silent state we can go to,
					# take into account just transition probability
					log_probability = pair_lse(log_probability,
						b[i*m + li] + self.out_transition_log_probabilities[l])

				# Now add this probability in with the probability accumulated
				# from transitions to subsequent non-silent states.
				b[i*m + k] = pair_lse(log_probability, b[i*m + k])

			for k in range(self.silent_start):
				# Do the non-silent states in the current step, which depend on
				# subsequent non-silent states and current-step silent states.

				# This holds the total accumulated log probability of going
				# to such states and continuing from there to the end.
				log_probability = NEGINF
				for l in range(out_edges[k], out_edges[k+1]):
					li = self.out_transitions[l]
					if li >= self.silent_start:
						continue

					# For each subsequent non-silent state l, take into account
					# transition and emission emission probability.
					log_probability = pair_lse(log_probability,
						b[(i+1)*m + li] + self.out_transition_log_probabilities[l] +
						e[i + li*n])

				for l in range(out_edges[k], out_edges[k+1]):
					li = self.out_transitions[l]
					if li < self.silent_start:
						continue

					# For each current-step silent state, add in the probability
					# of going from here to there and then continuing on to the
					# end of the sequence.
					log_probability = pair_lse(log_probability,
						b[i*m + li] + self.out_transition_log_probabilities[l])

				# Now we have summed the probabilities of all the ways we can
				# get from here to the end, so we can fill in the table entry.
				b[i*m + k] = log_probability

		if emissions is NULL:
			free(e)
		return b

	def fit(self, sequences, stop_threshold=1E-9,
		min_iterations=0, max_iterations=1e8, algorithm='baum-welch',
		pseudocount=None, transition_pseudocount=0, emission_pseudocount=0.0,
		use_pseudocount=False, inertia=None, edge_inertia=0.0,
		distribution_inertia=0.0, lr_decay=0.0):
		"""Fit the model to data using either Baum-Welch, Viterbi, or supervised training.

		Given a list of sequences, performs re-estimation on the model
		parameters. The two supported algorithms are "baum-welch", "viterbi",
		and "labeled", indicating their respective algorithm. "labeled"
		corresponds to supervised learning that requires passing in a matching
		list of labels for each symbol seen in the sequences.

		Training supports a wide variety of other options including using
		edge pseudocounts and either edge or distribution inertia.

		Parameters
		----------
		sequences : array-like
			An array of some sort (list, numpy.ndarray, tuple..) of sequences,
			where each sequence is a numpy array, which is 1 dimensional if
			the HMM is a one dimensional array, or multidimensional if the HMM
			supports multiple dimensions.

		stop_threshold : double, optional
			The threshold the improvement ratio of the models log probability
			in fitting the scores. Default is 1e-9.

		min_iterations : int, optional
			The minimum number of iterations to run Baum-Welch training for.
			Default is 0.

		max_iterations : int, optional
			The maximum number of iterations to run Baum-Welch training for.
			Default is 1e8.

		algorithm : 'baum-welch', 'viterbi', 'labeled'
			The training algorithm to use. Baum-Welch uses the forward-backward
			algorithm to train using a version of structured EM. Viterbi
			iteratively runs the sequences through the Viterbi algorithm and
			then uses hard assignments of observations to states using that.
			Default is 'baum-welch'. Labeled training requires that labels
			are provided for each observation in each sequence.

		pseudocount : double, optional
			A pseudocount to add to both transitions and emissions. If supplied,
			it will override both transition_pseudocount and emission_pseudocount
			in the same way that specifying `inertia` will override both
			`edge_inertia` and `distribution_inertia`. Default is None.

		transition_pseudocount : double, optional
			A pseudocount to add to all transitions to add a prior to the
			MLE estimate of the transition probability. Default is 0.

		emission_pseudocount : double, optional
			A pseudocount to add to the emission of each distribution. This
			effectively smoothes the states to prevent 0. probability symbols
			if they don't happen to occur in the data. Only effects hidden
			Markov models defined over discrete distributions. Default is 0.

		use_pseudocount : bool, optional
			Whether to use the pseudocounts defined in the `add_edge` method
			for edge-specific pseudocounts when updating the transition
			probability parameters. Does not effect the `transition_pseudocount`
			and `emission_pseudocount` parameters, but can be used in addition
			to them. Default is False.

		inertia : double or None, optional, range [0, 1]
			If double, will set both edge_inertia and distribution_inertia to
			be that value. If None, will not override those values. Default is
			None.

		edge_inertia : bool, optional, range [0, 1]
			Whether to use inertia when updating the transition probability
			parameters. Default is 0.0.

		distribution_inertia : double, optional, range [0, 1]
			Whether to use inertia when updating the distribution parameters.
			Default is 0.0.

		lr_decay : double, optional, positive
			The step size decay as a function of the number of iterations.
			Functionally, this sets the inertia to be (2+k)^{-lr_decay}
			where k is the number of iterations. This causes initial
			iterations to have more of an impact than later iterations,
			and is frequently used in minibatch learning. This value is
			suggested to be between 0.5 and 1. Default is 0, meaning no
			decay.

		multiple_check_input : bool, optional
			Whether to check and transcode input at each iteration. This
			leads to copying whole input data in each iteration. Which 
			can introduce significant overhead (up to 2 times slower) so 
			should be turned off when you know that data won't be changed 
			between fitting iteration. Default is True.

		Returns
		-------
		improvement : HiddenMarkovModel
			The trained model itself.
		"""

		if self.d == 0:
			raise ValueError("must bake model before fitting")

		cdef int iteration = 0
		cdef double improvement = INF
		cdef double initial_log_probability_sum
		cdef double log_probability_sum
		cdef double last_log_probability_sum

		if not isinstance(sequences, SequenceGenerator):
			data_generator = SequenceGenerator(sequences)
		else:
			data_generator = sequences

		while improvement > stop_threshold or iteration < min_iterations + 1:
			epoch_start_time = time.time()
			step_size = None if inertia is None else 1 - ((1 - inertia) * (2 + iteration) ** -lr_decay)

			self.from_summaries(step_size, pseudocount, transition_pseudocount,
				emission_pseudocount, use_pseudocount,
				edge_inertia, distribution_inertia)

			if iteration >= max_iterations + 1:
				break

			log_probability_sum = 0
			for batch in data_generator.batches():
				log_probability_sum += self.summarize(*batch)

			if iteration == 0:
				initial_log_probability_sum = log_probability_sum
			else:
				improvement = log_probability_sum - last_log_probability_sum

			iteration += 1
			last_log_probability_sum = log_probability_sum

		self.clear_summaries()
		return self

	def summarize(self, sequences, weights=None):
		"""Summarize data into stored sufficient statistics for out-of-core
		training. Only implemented for Baum-Welch training since Viterbi
		is less memory intensive.

		Parameters
		----------
		sequences : array-like
			An array of some sort (list, numpy.ndarray, tuple..) of sequences,
			where each sequence is a numpy array, which is 1 dimensional if
			the HMM is a one dimensional array, or multidimensional if the HMM
			supports multiple dimensions.

		Returns
		-------
		logp : double
			The log probability of the sequences.
		"""
		return sum([self._baum_welch_summarize(sequence, weight)
			for sequence, weight in zip(sequences, weights)])

	cpdef double _baum_welch_summarize(self, numpy.ndarray sequence_ndarray, double weight):
		"""Python wrapper for the summarization step.

		This is done to ensure compatibility with joblib's multithreading
		API. It just calls the cython update, but provides a Python wrapper
		which joblib can easily wrap.
		"""

		cdef double* sequence = <double*> sequence_ndarray.data
		cdef int n = sequence_ndarray.shape[0]
		cdef double log_sequence_probability

		with nogil:
			log_sequence_probability = self._summarize(sequence, &weight, n,
				0, self.d)

		return log_sequence_probability

	cdef double _summarize(self, double* sequence, double* weight, int n,
		int column_idx, int d) nogil:
		"""Collect sufficient statistics on a single sequence."""

		cdef int i, k, l, li
		cdef int m = self.n_states

		cdef void** distributions = self.distributions_ptr

		cdef double log_sequence_probability
		cdef double log_transition_emission_probability_sum

		cdef double* expected_transitions = <double*> calloc(self.n_edges, sizeof(double))
		cdef double* f
		cdef double* b
		cdef double* e

		cdef int* tied_edges = self.tied_edge_group_size
		cdef int* tied_states = self.tied_state_count
		cdef int* out_edges = self.out_edge_count

		cdef double* weights = <double*> calloc(n, sizeof(double))

		e = <double*> malloc(n*self.silent_start*sizeof(double))
		for l in range(self.silent_start):
			for i in range(n):
				with gil:
					python_log_probability(self.distributions[l], sequence+i*d, e+l*n+i, 1)

				e[l*n + i] += self.state_weights[l]

		f = self._forward(sequence, n, e)
		b = self._backward(sequence, n, e)

		log_sequence_probability = NEGINF
		for i in range(self.silent_start):
			log_sequence_probability = pair_lse(f[n*m + i],
				log_sequence_probability)

		# Is the sequence impossible? If so, we can't train on it, so skip
		# it
		if log_sequence_probability != NEGINF:
			for k in range(m):
				# For each state we could have come from
				for l in range(out_edges[k], out_edges[k+1]):
					li = self.out_transitions[l]
					if li >= self.silent_start:
						continue

					# For each state we could go to (and emit a character)
					# Sum up probabilities that we later normalize by
					# probability of sequence.
					log_transition_emission_probability_sum = NEGINF
					for i in range(n):
						# For each character in the sequence
						# Add probability that we start and get up to state k,
						# and go k->l, and emit the symbol from l, and go from l
						# to the end.
						log_transition_emission_probability_sum = pair_lse(
							log_transition_emission_probability_sum,
							f[i*m + k] +
							self.out_transition_log_probabilities[l] +
							e[i + li*n] + b[(i+1)*m + li])

					# Now divide by probability of the sequence to make it given
					# this sequence, and add as this sequence's contribution to
					# the expected transitions matrix's k, l entry.
					expected_transitions[l] += cexp(
						log_transition_emission_probability_sum -
						log_sequence_probability)

				for l in range(out_edges[k], out_edges[k+1]):
					li = self.out_transitions[l]
					if li < self.silent_start:
						continue
					# For each silent state we can go to on the same character
					# Sum up probabilities that we later normalize by
					# probability of sequence.
					log_transition_emission_probability_sum = NEGINF
					for i in range(n+1):
						# For each row in the forward DP table (where we can
						# have transitions to silent states) of which we have 1
						# more than we have symbols...

						# Add probability that we start and get up to state k,
						# and go k->l, and go from l to the end. In this case,
						# we use forward and backward entries from the same DP
						# table row, since no character is being emitted.
						log_transition_emission_probability_sum = pair_lse(
							log_transition_emission_probability_sum,
							f[i*m + k] + self.out_transition_log_probabilities[l]
							+ b[i*m + li])

					# Now divide by probability of the sequence to make it given
					# this sequence, and add as this sequence's contribution to
					# the expected transitions matrix's k, l entry.
					expected_transitions[l] += cexp(
						log_transition_emission_probability_sum -
						log_sequence_probability)

				if k < self.silent_start:
					for i in range(n):
						# For each symbol that came out
						# What's the weight of this symbol for that state?
						# Probability that we emit index characters and then
						# transition to state l, and that from state l we
						# continue on to emit len(sequence) - (index + 1)
						# characters, divided by the probability of the
						# sequence under the model.
						# According to http://www1.icsi.berkeley.edu/Speech/
						# docs/HTKBook/node7_mn.html, we really should divide by
						# sequence probability.
						weights[i] = cexp(f[(i+1)*m + k] + b[(i+1)*m + k] -
							log_sequence_probability) * weight[0]

					with gil:
						python_summarize(self.distributions[k], sequence,
							weights, n)

			# Update the master expected transitions vector representing the sparse matrix.
			with gil:
				for i in range(self.n_edges):
					self.expected_transitions[i] += expected_transitions[i] * weight[0]

		self.summaries += 1

		free(expected_transitions)
		free(e)
		free(weights)
		free(f)
		free(b)
		return log_sequence_probability * weight[0]

	def from_summaries(self, inertia=None, pseudocount=None,
		transition_pseudocount=0.0, emission_pseudocount=0.0,
		use_pseudocount=False, edge_inertia=0.0, distribution_inertia=0.0):
		"""Fit the model to the stored summary statistics.

		Parameters
		----------
		inertia : double or None, optional
			The inertia to use for both edges and distributions without
			needing to set both of them. If None, use the values passed
			in to those variables. Default is None.

		pseudocount : double, optional
			A pseudocount to add to both transitions and emissions. If supplied,
			it will override both transition_pseudocount and emission_pseudocount
			in the same way that specifying `inertia` will override both
			`edge_inertia` and `distribution_inertia`. Default is None.

		transition_pseudocount : double, optional
			A pseudocount to add to all transitions to add a prior to the
			MLE estimate of the transition probability. Default is 0.

		emission_pseudocount : double, optional
			A pseudocount to add to the emission of each distribution. This
			effectively smoothes the states to prevent 0. probability symbols
			if they don't happen to occur in the data. Only effects hidden
			Markov models defined over discrete distributions. Default is 0.

		use_pseudocount : bool, optional
			Whether to use the pseudocounts defined in the `add_edge` method
			for edge-specific pseudocounts when updating the transition
			probability parameters. Does not effect the `transition_pseudocount`
			and `emission_pseudocount` parameters, but can be used in addition
			to them. Default is False.

		edge_inertia : bool, optional, range [0, 1]
			Whether to use inertia when updating the transition probability
			parameters. Default is 0.0.

		distribution_inertia : double, optional, range [0, 1]
			Whether to use inertia when updating the distribution parameters.
			Default is 0.0.

		Returns
		-------
		None
		"""

		if self.d == 0:
			raise ValueError("must bake model before using from summaries")

		if self.summaries == 0:
			return

		if inertia is not None:
			edge_inertia = inertia
			distribution_inertia = inertia

		if pseudocount is not None:
			transition_pseudocount = pseudocount
			emission_pseudocount = pseudocount

		self._from_summaries(transition_pseudocount, emission_pseudocount,
			use_pseudocount, edge_inertia, distribution_inertia)

		memset(self.expected_transitions, 0, self.n_edges*sizeof(double))
		self.summaries = 0

	cdef void _from_summaries(self, double transition_pseudocount,
		double emission_pseudocount, bint use_pseudocount, double edge_inertia,
		double distribution_inertia):
		"""Update the transition matrix and emission distributions."""

		cdef int k, i, l, li, m = len(self.states), n, idx
		cdef int* in_edges = self.in_edge_count
		cdef int* out_edges = self.out_edge_count

		cdef int* tied_states = self.tied_state_count
		cdef double* norm

		cdef double probability, tied_edge_probability
		cdef int start, end
		cdef int* tied_edges = self.tied_edge_group_size

		cdef double* expected_transitions = <double*> calloc(m*m, sizeof(double))

		with nogil:
			for k in range(m):
				for l in range(out_edges[k], out_edges[k+1]):
					li = self.out_transitions[l]
					expected_transitions[k*m + li] = self.expected_transitions[l]

			# We now have expected_transitions taking into account all sequences.
			# And a list of all emissions, and a weighting of each emission for each
			# state
			# Normalize transition expectations per row (so it becomes transition
			# probabilities)
			# See http://stackoverflow.com/a/8904762/402891
			# Only modifies transitions for states a transition was observed from.
			norm = <double*> calloc(m, sizeof(double))

			# Go through the tied state groups and add transitions from each member
			# in the group to the other members of the group.
			# For each group defined.
			for k in range(self.n_tied_edge_groups-1):
				tied_edge_probability = 0.

				# For edge in this group, get the sum of the edges
				for l in range(tied_edges[k], tied_edges[k+1]):
					start = self.tied_edges_starts[l]
					end = self.tied_edges_ends[l]
					tied_edge_probability += expected_transitions[start*m + end]

				# Update each entry
				for l in range(tied_edges[k], tied_edges[k+1]):
					start = self.tied_edges_starts[l]
					end = self.tied_edges_ends[l]
					expected_transitions[start*m + end] = tied_edge_probability

			# Calculate the regularizing norm for each node
			for k in range(m):
				for l in range(out_edges[k], out_edges[k+1]):
					li = self.out_transitions[l]
					norm[k] += expected_transitions[k*m + li] + \
						transition_pseudocount + \
						self.out_transition_pseudocounts[l] * use_pseudocount

			# For every node, update the transitions appropriately
			for k in range(m):
				# Recalculate each transition out from that node and update
				# the vector of out transitions appropriately
				if norm[k] > 0:
					for l in range(out_edges[k], out_edges[k+1]):
						li = self.out_transitions[l]
						probability = (expected_transitions[k*m + li] +
							transition_pseudocount +
							self.out_transition_pseudocounts[l] * use_pseudocount)\
							/ norm[k]
						self.out_transition_log_probabilities[l] = _log(
							cexp(self.out_transition_log_probabilities[l]) *
							edge_inertia + probability * (1 - edge_inertia))

				# Recalculate each transition in to that node and update the
				# vector of in transitions appropriately
				for l in range(in_edges[k], in_edges[k+1]):
					li = self.in_transitions[l]
					if norm[li] > 0:
						probability = (expected_transitions[li*m + k] +
							transition_pseudocount +
							self.in_transition_pseudocounts[l] * use_pseudocount)\
							/ norm[li]
						self.in_transition_log_probabilities[l] = _log(
							cexp(self.in_transition_log_probabilities[l]) *
							edge_inertia + probability * (1 - edge_inertia))

			for k in range(self.silent_start):
				# Re-estimate the emission distribution for every non-silent state.
				# Take each emission weighted by the probability that we were in
				# this state when it came out, given that the model generated the
				# sequence that the symbol was part of. Take into account tied
				# states by only training that distribution one time, since many
				# states are pointing to the same distribution object.
				with gil:
					self.states[k].distribution.from_summaries(distribution_inertia)

		for k in range(self.n_states):
			for l in range(self.out_edge_count[k], self.out_edge_count[k+1]):
				li = self.out_transitions[l]
				prob = self.out_transition_log_probabilities[l]
				self.graph[self.states[k]][self.states[li]]['probability'] = prob

		free(norm)
		free(expected_transitions)

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object.

		Parameters
		----------
		None

		Returns
		-------
		None
		"""

		memset(self.expected_transitions, 0, self.n_edges*sizeof(double))
		self.summaries = 0

		for state in self.states[:self.silent_start]:
			state.distribution.clear_summaries()

	@classmethod
	def from_matrix(cls, transition_probabilities, distributions, starts):
		"""Create a model from a more standard matrix format.

		Take in a 2D matrix of floats of size n by n, which are the transition
		probabilities to go from any state to any other state. May also take in
		a list of length n representing the names of these nodes, and a model
		name. Must provide the matrix, and a list of size n representing the
		distribution you wish to use for that state, a list of size n indicating
		the probability of starting in a state, and a list of size n indicating
		the probability of ending in a state.

		Parameters
		----------
		transition_probabilities : array-like, shape (n_states, n_states)
			The probabilities of each state transitioning to each other state.

		distributions : array-like, shape (n_states)
			The distributions for each state. Silent states are indicated by
			using None instead of a distribution object.

		starts : array-like, shape (n_states)
			The probabilities of starting in each of the states.

		Returns
		-------
		model : HiddenMarkovModel
			The baked model ready to go.

		Examples
		--------
		matrix = [[0.4, 0.5], [0.4, 0.5]]
		distributions = [NormalDistribution(1, .5), NormalDistribution(5, 2)]
		starts = [1., 0.]
		ends = [.1., .1]
		state_names= ["A", "B"]

		model = Model.from_matrix(matrix, distributions, starts, ends,
			state_names, name="test_model")
		"""

		# Build the initial model
		model = HiddenMarkovModel()
		state_names = ["s{}".format(i) for i in range(len(distributions))]

		# Build state objects for every state with the appropriate distribution
		states = [State(distribution, name=name) for name, distribution in
			zip(state_names, distributions)]

		n = len(states)

		# Add all the states to the model
		for state in states:
			model.add_state(state)

		# Connect the start of the model to the appropriate state
		for i, prob in enumerate(starts):
			if prob != 0:
				model.add_transition(model.start, states[i], prob)

		# Connect all states to each other if they have a non-zero probability
		for i in range(n):
			for j, prob in enumerate(transition_probabilities[i]):
				if prob != 0.:
					model.add_transition(states[i], states[j], prob)

		model.bake()
		return model

	@classmethod
	def init_params(cls, n_components, X):
		"""Create HHM model based on number of states and sample data.

		Parameters
		----------

		n_components : int
			number of states

		X : array
			sample data

		Returns
		-------
		initialized model
		"""

		data_for_clustering = [item for item in X]
		data_for_clustering = numpy.array(data_for_clustering)
		data_for_clustering = numpy.concatenate(data_for_clustering).reshape(-1, 1)

		kmeans = KMeans(n_clusters=n_components, random_state=0)
		y_means = kmeans.fit_predict(data_for_clustering)

		X_ = [data_for_clustering[y_means == i] for i in range(n_components)]

		distributions = []
		for i in range(n_components):
			dist = NormalDistribution.blank()
			dist.fit(X_[i])
			distributions.append(dist)

		k = n_components
		transition_matrix = numpy.ones((k, k)) / k
		start_probabilities = numpy.zeros(k)
		start_probabilities[0] = 1

		model = HiddenMarkovModel.from_matrix(transition_matrix, distributions, start_probabilities)

		return model
