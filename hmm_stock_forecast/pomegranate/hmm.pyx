#cython: boundscheck=False
#cython: cdivision=True
# hmm.pyx: Yet Another Hidden Markov Model library
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
#          Adam Novak <anovak1@ucsc.edu>

from __future__ import print_function

from libc.math cimport exp as cexp
from operator import attrgetter
import math
import networkx
import tempfile
import warnings
import time

from .base cimport Model
from .base cimport State

from distributions.distributions cimport Distribution

from sklearn.cluster import KMeans

from .utils cimport _log
from .utils cimport pair_lse
from .utils cimport python_log_probability
from .utils cimport python_summarize

from .utils import check_random_state
from .utils import _check_nan

from .io import BaseGenerator
from .io import SequenceGenerator

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.string cimport memset

import numpy
cimport numpy

from joblib import Parallel
from joblib import delayed

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463

def _check_input(sequence, model):
	n = len(sequence)

	if not isinstance(model, Distribution) and not isinstance(model, Model):
		return numpy.array(sequence, dtype=numpy.float64)

	if not model.discrete:
		sequence_ndarray = numpy.array(sequence, dtype=numpy.float64)

	elif model.multivariate and model.discrete:
		sequence_ndarray = numpy.empty((n, model.d), dtype=numpy.float64)

		for i in range(n):
			for j in range(model.d):
				symbol = sequence[i][j]
				keymap = model.keymap[j]

				if _check_nan(symbol):
					sequence_ndarray[i, j] = numpy.nan
				elif symbol in keymap:
					sequence_ndarray[i, j] = keymap[symbol]
				else:
					raise ValueError("Symbol '{}' is not defined in a distribution"
						.format(symbol))
	else:
		sequence_ndarray = numpy.empty(n, dtype=numpy.float64)
		keymap = model.keymap[0]

		for i in range(n):
			symbol = sequence[i]

			if _check_nan(symbol):
				sequence_ndarray[i] = numpy.nan
			elif sequence[i] in keymap:
				sequence_ndarray[i] = keymap[symbol]
			else:
				raise ValueError("Symbol '{}' is not defined in a distribution"
					.format(symbol))

	return sequence_ndarray

def _initialize_distributions(X, distribution):
	"""This function initializes the distribution to labeled data.

	At the beginning of training, the distributions in an HMM must be
	initialized. Because of the flexibility of how these distributions
	can be specified, the specific user inputs must be decoded into
	the precise probability distributions they intended, and these
	distributions must be fit to the data. This function takes in
	data that has been labeled, either through user-specified labels
	or through a clustering algorithm run on unlabeled data, and
	decodes the user input and fits that to the data using a MLE
	fit.

	Parameters
	----------
	X: list
		A list of data partitions, where each partition is the data
		associated with one component of the model.

	distribution: flexible
		The user input for the type of probability distribution to
		fit.

	Returns
	-------
	distributons: list
		A list of distributions initialized to the given data.
	"""

	print('_initialize_distributions #1')
	n_components = len(X)
	d = X[0].shape[1]
	distributions = [] 

	if callable(distribution):
		print('_initialize_distributions #2')
		for i in range(n_components):
			if d == 1:
				print('_initialize_distributions #3')
				dist = distribution.from_samples(X[i])
			else:
				print('_initialize_distributions #4')
				dist = distribution.from_samples(X[i])
			#else:
			#	dist = IndependentComponentsDistribution.from_samples(
			#		X[i], distributions=distribution)
			
			distributions.append(dist)

	else:
		raise ValueError("distribution must be either a callable (e.g., " /
			"NormalDistribution) or a list of callables.")

	return distributions


cdef class HiddenMarkovModel(Model):
	"""A Hidden Markov Model

	A Hidden Markov Model (HMM) is a directed graphical model where nodes are
	hidden states which contain an observed emission distribution and edges
	contain the probability of transitioning from one hidden state to another.
	HMMs allow you to tag each observation in a variable length sequence with
	the most likely hidden state according to the model.

	Parameters
	----------
	name : str, optional
		The name of the model. Default is None.

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

	end_index : int
		The index of the end object in the state list

	silent_start : int
		The index of the beginning of the silent states in the state list

	states : list
		The list of all states in the model, with silent states at the end

	Examples
	--------
	"""

	cdef public object start, end
	cdef public int start_index
	cdef public int end_index
	cdef public int silent_start
	cdef double* in_transition_pseudocounts
	cdef double* out_transition_pseudocounts
	cdef double [:] state_weights
	cdef public bint discrete
	cdef public bint multivariate
	cdef int summaries
	cdef int cython
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
	cdef int finite, n_tied_edge_groups
	cdef public list keymap
	cdef object state_names
	cdef dict state_name_mapping
	cdef numpy.ndarray distributions
	cdef void** distributions_ptr

	def __init__(self, name=None, start=None, end=None):
		# Save the name or make up a name.
		self.name = str(name) or str(id(self))
		self.model = "HiddenMarkovModel"

		# This holds a directed graph between states. Nodes in that graph are
		# State objects, so they're guaranteed never to conflict when composing
		# two distinct models
		self.graph = networkx.DiGraph()

		# Save the start and end or mae one up
		self.start = start or State(None, name=self.name + "-start")
		self.end = end or State(None, name=self.name + "-end")

		# Put start and end in the graph
		self.graph.add_node(self.start)
		self.graph.add_node(self.end)

		self.state_names = set()
		self.state_name_mapping = {}

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
		"""Add a state to the given model.

		The state must not already be in the model, nor may it be part of any
		other model that will eventually be combined with this one.

		Parameters
		----------
		state : State
			A state object to be added to the model.

		Returns
		-------
		None
		"""

		if state.name in self.state_names:
			raise ValueError("A state with name '{}' already exists".format(state.name))

		self.graph.add_node(state)
		self.state_names.add(state.name)

	def add_states(self, *states):
		"""Add multiple states to the model at the same time.

		Parameters
		----------
		states : list or generator
			Either a list of states which are entered sequentially, or just
			comma separated values, for example model.add_states(a, b, c, d).

		Returns
		-------
		None
		"""

		for state in states:
			if isinstance(state, list):
				for s in state:
					self.add_state(s)
			else:
				self.add_state(state)

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

	def bake(self, verbose=False, merge="all"):
		"""Finalize the topology of the model.

		Finalize the topology of the model and assign a numerical index to
		every state. This method must be called before any of the probability-
		calculating methods.

		This fills in self.states (a list of all states in order) and
		self.transition_log_probabilities (log probabilities for transitions),
		as well as self.start_index and self.end_index, and self.silent_start
		(the index of the first silent state).

		Parameters
		----------
		verbose : bool, optional
			Return a log of changes made to the model during normalization
			or merging. Default is False.

		merge : "None", "Partial, "All"
			Merging has three options:
			"None": No modifications will be made to the model.
			"Partial": A silent state which only has a probability 1 transition
				to another silent state will be merged with that silent state.
				This means that if silent state "S1" has a single transition
				to silent state "S2", that all transitions to S1 will now go
				to S2, with the same probability as before, and S1 will be
				removed from the model.
			"All": A silent state with a probability 1 transition to any other
				state, silent or symbol emitting, will be merged in the manner
				described above. In addition, any orphan states will be removed
				from the model. An orphan state is a state which does not have
				any transitions to it OR does not have any transitions from it,
				except for the start and end of the model. This will iteratively
				remove orphan chains from the model. This is sometimes desirable,
				as all states should have both a transition in to get to that
				state, and a transition out, even if it is only to itself. If
				the state does not have either, the HMM will likely not work as
				intended.
			Default is 'All'.

		Returns
		-------
		None
		"""

		if len(self.graph.nodes()) == 0:
			raise ValueError("Must add states to the model before baking it.")

		cdef int i

		self.free_bake_buffers()

		in_edge_count = numpy.zeros(len(self.graph.nodes()),
			dtype=numpy.int32)
		out_edge_count = numpy.zeros(len(self.graph.nodes()),
			dtype=numpy.int32)

		merge = merge.lower() if merge else None
		while merge == 'all':
			merge_count = 0

			# Reindex the states based on ones which are still there
			prestates = list(self.graph.nodes)
			indices = { prestates[i]: i for i in range(len(prestates)) }

			# Go through all the edges, summing in and out edges
			for a, b in list(self.graph.edges()):
				out_edge_count[indices[a]] += 1
				in_edge_count[indices[b]] += 1

			# Go through each state, and if either in or out edges are 0,
			# remove the edge.
			for i in range(len(prestates)):
				if prestates[i] is self.start or prestates[i] is self.end:
					continue

				if in_edge_count[i] == 0:
					merge_count += 1
					self.graph.remove_node(prestates[i])

					if verbose:
						print("Orphan state {} removed due to no edges \
							leading to it".format(prestates[i].name))

				elif out_edge_count[i] == 0:
					merge_count += 1
					self.graph.remove_node(prestates[i])

					if verbose:
						print("Orphan state {} removed due to no edges \
							leaving it".format(prestates[i].name))

			if merge_count == 0:
				break

		# Go through the model checking to make sure out edges sum to 1.
		# Normalize them to 1 if this is not the case.
		if merge in ['all', 'partial']:
			for state in list(self.graph.nodes()):

				# Perform log sum exp on the edges to see if they properly sum to 1
				out_edges = round(sum(numpy.e**x['probability']
					for x in self.graph.adj[state].values()), 8)

				# The end state has no out edges, so will be 0
				if out_edges != 1. and state != self.end:
					# Issue a notice if verbose is activated
					if verbose:
						print("{} : {} summed to {}, normalized to 1.0"\
							.format(self.name, state.name, out_edges))

					# Reweight the edges so that the probability (not logp) sums
					# to 1.
					for edge in self.graph.adj[state].values():
						edge['probability'] = edge['probability'] - _log(out_edges)

		# Automatically merge adjacent silent states attached by a single edge
		# of 1.0 probability, as that adds nothing to the model. Traverse the
		# edges looking for 1.0 probability edges between silent states.
		while merge in ['all', 'partial']:
			# Repeatedly go through the model until no merges take place.
			merge_count = 0

			for a, b, e in list(self.graph.edges(data=True)):
				# Since we may have removed a or b in a previous iteration,
				# a simple fix is to just check to see if it's still there
				if a not in list(self.graph.nodes()) or b not in list(self.graph.nodes()):
					continue

				if a == self.start or b == self.end:
					continue

				# If a silent state has a probability 1 transition out
				if e['probability'] == 0.0 and a.is_silent():

					# Make sure the transition is an appropriate merger
					if merge=='all' or (merge=='partial' and b.is_silent()):

						# Go through every transition to that state
						for x, y, d in self.graph.edges(data=True):

							# Make sure that the edge points to the current node
							if y is a:
								# Increment the edge counter
								merge_count += 1

								# Remove the edge going to that node
								self.graph.remove_edge(x, y)

								pseudo = max(e['pseudocount'], d['pseudocount'])
								group = e['group'] if e['group'] == d['group'] else None
								# Add a new edge going to the new node
								self.graph.add_edge(x, b, probability=d['probability'],
									pseudocount=pseudo,
									group=group)

								# Log the event
								if verbose:
									print("{} : {} - {} merged".format(
										self.name, a, b))

						# Remove the state now that all edges are removed
						self.graph.remove_node(a)

			if merge_count == 0:
				break

		if merge in ['all', 'partial']:
			# Detect whether or not there are loops of silent states by going
			# through every pair of edges, and ensure that there is not a cycle
			# of silent states.
			for a, b, e in self.graph.edges(data=True):
				for x, y, d in self.graph.edges(data=True):
					if a is y and b is x and a.is_silent() and b.is_silent():
						print("Loop: {} - {}".format(a.name, b.name))

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

		# Determine if the model is infinite or not based on the number of edges
		# to the end state
		if self.in_edge_count[indices[self.end]+1] == 0:
			self.finite = 0
		else:
			self.finite = 1
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
		self.multivariate = self.d > 1

		if self.d > 1:
			keys = [[] for i in range(self.d)]
			self.keymap = [{} for i in range(self.d)]
			for state in self.states[:self.silent_start]:
				d = state.distribution

			for i in range(self.d):
				keys[i] = tuple(set(keys[i]))
				self.keymap[i] = {key: j for j, key in enumerate(keys[i])}

			for state in self.states[:self.silent_start]:
				d = state.distribution

		self.state_name_mapping = {state.name: i for i, state in enumerate(self.states)}
		self.distributions = numpy.empty(self.silent_start, dtype='object')
		for i in range(self.silent_start):
			self.distributions[i] = self.states[i].distribution
			if self.d != self.distributions[i].d:
				raise ValueError("mis-matching inputs for states")

		self.distributions_ptr = <void**> self.distributions.data

		self.cython = 1
		for dist in self.distributions:
			if not isinstance(dist, Distribution) and not isinstance(dist, Model):
				self.cython = 0

		# This holds the index of the start state
		try:
			self.start_index = indices[self.start]
		except KeyError:
			raise SyntaxError("Model.start has been deleted, leaving the \
				model with no start. Please ensure it has a start.")
		# And the end state
		try:
			self.end_index = indices[self.end]
		except KeyError:
			raise SyntaxError("Model.end has been deleted, leaving the \
				model with no end. Please ensure it has an end.")


	def sample(self, n=None, length=0, path=False, random_state=None):
		"""Generate a sequence from the model.

		Returns the sequence generated, as a list of emitted items. The
		model must have been baked first in order to run this method.

		If a length is specified and the HMM is infinite (no edges to the
		end state), then that number of samples will be randomly generated.
		If the length is specified and the HMM is finite, the method will
		attempt to generate a prefix of that length. Currently it will force
		itself to not take an end transition unless that is the only path,
		making it not a true random sample on a finite model.

		WARNING: If the HMM has no explicit end state, must specify a length
		to use.

		Parameters
		----------
		n : int or None, optional
			The number of samples to generate. If None, return only one sample.

		length : int, optional
			Generate a sequence with a maximal length of this size. Used if
			you have no explicit end state. Default is 0.

		path : bool, optional
			Return the path of hidden states in addition to the emissions. If
			true will return a tuple of (sample, path). Default is False.

		random_state : int, numpy.random.RandomState, or None
			The random state used for generating samples. If set to none, a
			random seed will be used. If set to either an integer or a
			random seed, will produce deterministic outputs.

		Returns
		-------
		sample : list or tuple
			If path is true, return a tuple of (sample, path), otherwise return
			just the samples.
		"""

		if self.d == 0:
			raise ValueError("must bake model before sampling")

		random_state = check_random_state(random_state)

		if n is None:
			return self._sample(length, path, random_state)
		else:
			return [self.sample(length=length, path=path, random_state=i) 
				for i in random_state.randint(10000000, size=n)]


	cdef numpy.ndarray _sample(self, int length, int path, random_state):
		cdef int i, j, k, l, li, m=len(self.states)
		cdef double cumulative_probability
		cdef double [:,:] transition_probabilities = numpy.zeros((m,m))
		cdef double [:] cum_probabilities = numpy.zeros(self.n_edges)

		cdef int*  out_edges = self.out_edge_count

		for k in range(m):
			cumulative_probability = 0.
			for l in range(out_edges[k], out_edges[k+1]):
				cumulative_probability += cexp(
					self.out_transition_log_probabilities[l])
				cum_probabilities[l] = cumulative_probability

		# This holds the numerical index of the state we are currently in.
		# Start in the start state
		i = self.start_index

		# Record the number of samples
		cdef int n = 0
		# Define the list of emissions, and the path of hidden states taken
		cdef list emissions = [], sequence_path = []
		cdef State state
		cdef double sample

		while i != self.end_index:
			# Get the object associated with this state
			state = self.states[i]

			# Add the state to the growing path
			sequence_path.append(state)

			if not state.is_silent():
				# There's an emission distribution, so sample from it
				emissions.append(state.distribution.sample(
					random_state=random_state))
				n += 1

			# If we've reached the specified length, return the appropriate
			# values
			if length != 0 and n >= length:
				if path:
					return numpy.array([emissions, sequence_path])
				return numpy.array(emissions)

			# What should we pick as our next state?
			# Generate a number between 0 and 1 to make a weighted decision
			# as to which state to jump to next.
			sample = random_state.uniform(0, 1)

			# Save the last state id we were in
			j = i

			# Find out which state we're supposed to go to by comparing the
			# random number to the list of cumulative probabilities for that
			# state, and then picking the selected state.
			for k in range(out_edges[i], out_edges[i+1]):
				if cum_probabilities[k] > sample:
					i = self.out_transitions[k]
					break

			# If the user specified a length, and we're not at that length, and
			# we're in an infinite HMM, we want to avoid going to the end state
			# if possible. If there is only a single probability 1 end to the
			# end state we can't avoid it, otherwise go somewhere else.
			if length != 0 and self.finite == 1 and i == self.end_index:
				# If there is only one transition...
				if len(range(out_edges[j], out_edges[j+1])) == 1:
					# ...and that transition goes to the end of the model...
					if self.out_transitions[out_edges[j]] == self.end_index:
						# ... then end the sampling, as nowhere else to go.
						break

				# Take the cumulative probability of not going to the end state
				cumulative_probability = 0.
				for k in range(out_edges[k], out_edges[k+1]):
					if self.out_transitions[k] != self.end_index:
						cumulative_probability += cum_probabilities[k]

				# Randomly select a number in that probability range
				sample = random_state.uniform(0, cumulative_probability)

				# Select the state is corresponds to
				for k in range(out_edges[i], out_edges[i+1]):
					if cum_probabilities[k] > sample:
						i = self.out_transitions[k]
						break

		# Done! Return either emissions, or emissions and path.
		if path:
			sequence_path.append(self.end)
			return numpy.array([emissions, sequence_path])

		return numpy.array(emissions)

	cpdef double log_probability(self, sequence, check_input=True):
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
		cdef int mv = self.multivariate

		if check_input:
			sequence_ndarray = _check_input(sequence, self)
		else:
			sequence_ndarray = sequence

		sequence_ptr = <double*> sequence_ndarray.data

		with nogil:
			log_probability = self._vl_log_probability(sequence_ptr, n)

		return log_probability

	cdef double _vl_log_probability(self, double* sequence, int n) nogil:
		cdef double* f = self._forward(sequence, n, NULL)
		cdef double log_probability
		cdef int i, m = self.n_states

		if self.finite == 1:
			log_probability = f[n*m + self.end_index]
		else:
			log_probability = NEGINF
			for i in range(self.silent_start):
				log_probability = pair_lse(log_probability, f[n*m + i])

		free(f)
		return log_probability

	cpdef numpy.ndarray forward(self, sequence):
		"""Run the forward algorithm on the sequence.

		Calculate the probability of each observation being aligned to each
		state by going forward through a sequence. Returns the full forward
		matrix. Each index i, j corresponds to the sum-of-all-paths log
		probability of starting at the beginning of the sequence, and aligning
		observations to hidden states in such a manner that observation i was
		aligned to hidden state j. Uses row normalization to dynamically scale
		each row to prevent underflow errors.

		If the sequence is impossible, will return a matrix of nans.

		See also:
			- Silent state handling taken from p. 71 of "Biological
		Sequence Analysis" by Durbin et al., and works for anything which
		does not have loops of silent states.
			- Row normalization technique explained by
		http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf on p. 14.

		Parameters
		----------
		sequence : array-like
			An array (or list) of observations.

		Returns
		-------
		matrix : array-like, shape (len(sequence), n_states)
			The probability of aligning the sequences to states in a forward
			fashion.
		"""

		if self.d == 0:
			raise ValueError("must bake model before using forward algorithm")

		cdef numpy.ndarray sequence_ndarray
		cdef double* sequence_data
		cdef int n = len(sequence), m = len(self.states)
		cdef int mv = self.multivariate
		cdef void** distributions = <void**> self.distributions.data
		cdef numpy.ndarray f_ndarray = numpy.empty((n+1, m), dtype=numpy.float64)
		cdef double* f

		sequence_ndarray = _check_input(sequence, self)
		sequence_data = <double*> sequence_ndarray.data

		with nogil:
			f = <double*> self._forward(sequence_data, n, NULL)

		for i in range(n+1):
			for j in range(m):
				f_ndarray[i, j] = f[i*m + j]

		free(f)
		return f_ndarray

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
		if self.finite == 1:
			for i in range(m):
				b[n*m + i] = NEGINF
			b[n*m + self.end_index] = 0
		else:
			for i in range(self.silent_start):
				b[n*m + i] = 0.
			for i in range(self.silent_start, m):
				b[n*m + i] = NEGINF

		for kr in range(m-self.silent_start):
			if self.finite == 0:
				break
			# Cython arrays cannot go backwards, so modify the loop to account
			# for this.
			k = m - kr - 1

			# Do the silent states' dependencies on each other.
			# Doing it in reverse order ensures that anything we can
			# possibly transition to is already done.

			if k == self.end_index:
				# We already set the log-probability for this, so skip it
				continue

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
					b[n*m + li] + self.out_transition_log_probabilities[l])

			# Now this is the probability of reaching the end state given we are
			# in this silent state.
			b[n*m + k] = log_probability

		for k in range(self.silent_start):
			if self.finite == 0:
				break
			# Do the non-silent states in the last step, which depend on
			# current-step silent states.

			# This holds the total accumulated log probability of going
			# to such states and continuing from there to the end.
			log_probability = NEGINF
			for l in range(out_edges[k], out_edges[k+1]):
				li = self.out_transitions[l]
				if li < self.silent_start:
					continue

				# For each current-step silent state, add in the probability
				# of going from here to there and then continuing on to the
				# end of the sequence.
				log_probability = pair_lse(log_probability,
					b[n*m + li] + self.out_transition_log_probabilities[l])

			# Now we have summed the probabilities of all the ways we can
			# get from here to the end, so we can fill in the table entry.
			b[n*m + k] = log_probability

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

	def predict_log_proba(self, sequence):
		"""Calculate the state log probabilities for each observation in the sequence.

		Run the forward-backward algorithm on the sequence and return the emission
		matrix. This is the log normalized probability that each each state
		generated that emission given both the symbol and the entire sequence.

		This is a sklearn wrapper for the forward backward algorithm.

		See also:
			- Forward and backward algorithm implementations. A comprehensive
			description of the forward, backward, and forward-background
			algorithm is here:
			http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm

		Parameters
		----------
		sequence : array-like
			An array (or list) of observations.

		Returns
		-------
		emissions : array-like, shape (len(sequence), n_nonsilent_states)
			The log normalized probabilities of each state generating each emission.
		"""

		if self.d == 0:
			raise ValueError("must bake model before prediction")

		cdef int n = len(sequence), m = len(self.states)
		cdef int mv = self.multivariate
		cdef numpy.ndarray sequence_ndarray
		cdef numpy.ndarray r_ndarray = numpy.zeros((n, self.silent_start), dtype='float64')
		cdef double* sequence_data
		cdef double* r = <double*> r_ndarray.data

		sequence_ndarray = _check_input(sequence, self)
		sequence_data = <double*> sequence_ndarray.data

		with nogil:
			self._predict_log_proba(sequence_data, r, n, NULL)

		return r_ndarray

	cdef void _predict_log_proba(self, double* sequence, double* r, int n,
		double* emissions) nogil:
		cdef int i, k, l, li
		cdef int m = self.n_states, dim = self.d
		cdef double log_sequence_probability
		cdef double* f
		cdef double* b
		cdef double* e
		cdef void** distributions = self.distributions_ptr

		if emissions is NULL:
			e = <double*> malloc(n*self.silent_start*sizeof(double))
			for l in range(self.silent_start):
				for i in range(n):
					with gil:
						python_log_probability(self.distributions[l], sequence+i*dim, e+l*n+i, 1)

					e[l*n + i] += self.state_weights[l]
		else:
			e = emissions

		# Fill in both the F and B DP matrices.
		f = self._forward(sequence, n, emissions)
		b = self._backward(sequence, n, emissions)

		# Find out the probability of the sequence
		if self.finite == 1:
			log_sequence_probability = f[n*m + self.end_index]
		else:
			log_sequence_probability = NEGINF
			for i in range(self.silent_start):
				log_sequence_probability = pair_lse(
					log_sequence_probability, f[n*m + i])

		# Is the sequence impossible? If so, don't bother calculating any more.
		if log_sequence_probability == NEGINF:
			with gil:
				print("Warning: Sequence is impossible.")

		for k in range(m):
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
					r[i*self.silent_start + k] = f[(i+1)*m + k] + b[(i+1)*m + k] - \
						log_sequence_probability

		free(f)
		free(b)
		free(e)

	def predict(self, sequence, algorithm='map'):
		"""Calculate the most likely state for each observation.

		This can be either the Viterbi algorithm or maximum a posteriori. It
		returns the probability of the sequence under that state sequence and
		the actual state sequence.

		This is a sklearn wrapper for the Viterbi and maximum_a_posteriori methods.

		Parameters
		----------
		sequence : array-like
			An array (or list) of observations.

		algorithm : "map", "viterbi"
			The algorithm with which to decode the sequence

		Returns
		-------
		path : list of integers
			A list of the ids of states along the MAP or the Viterbi path.
		"""

		if self.d == 0:
			raise ValueError("must bake model before prediction")
		
		if algorithm == 'map':
			return [state_id for state_id, state in self.maximum_a_posteriori(sequence)[1]]
		else:
			raise ValueError("algorithm must be map or viterbi")

	def maximum_a_posteriori(self, sequence):
		"""Run posterior decoding on the sequence.

		MAP decoding is an alternative to viterbi decoding, which returns the
		most likely state for each observation, based on the forward-backward
		algorithm. This is also called posterior decoding. This method is
		described on p. 14 of http://ai.stanford.edu/~serafim/CS262_2007/
		notes/lecture5.pdf

		WARNING: This may produce impossible sequences.

		Parameters
		----------
		sequence : array-like
			An array (or list) of observations.

		Returns
		-------
		logp : double
			The log probability of the sequence under the Viterbi path

		path : list of tuples
			Tuples of (state index, state object) of the states along the
			posterior path.
		"""

		if self.d == 0:
			raise ValueError("must bake model before using MAP decoding")

		return self._maximum_a_posteriori(numpy.array(sequence))


	cdef tuple _maximum_a_posteriori(self, numpy.ndarray sequence):
		cdef int i, k, l, li
		cdef int m=len(self.states), n=len(sequence)
		cdef double [:,:] emission_weights = self.predict_log_proba(sequence)

		cdef list path = []
		cdef double maximum_emission_weight
		cdef double log_probability_sum = 0
		cdef int maximum_index

		# Go through each symbol and determine what the most likely state
		# that it came from is.
		for k in range(n):
			maximum_index = -1
			maximum_emission_weight = NEGINF

			# Go through each hidden state and see which one has the maximal
			# weight for emissions. Tied states are not taken into account
			# here, because we are not performing training.
			for l in range(self.silent_start):
				if emission_weights[k, l] > maximum_emission_weight:
					maximum_emission_weight = emission_weights[k, l]
					maximum_index = l

			path.append((maximum_index, self.states[maximum_index]))
			log_probability_sum += maximum_emission_weight

		return log_probability_sum, path

	def fit(self, sequences, weights=None, labels=None, stop_threshold=1E-9,
		min_iterations=0, max_iterations=1e8, algorithm='baum-welch',
		pseudocount=None, transition_pseudocount=0, emission_pseudocount=0.0,
		use_pseudocount=False, inertia=None, edge_inertia=0.0,
		distribution_inertia=0.0, batches_per_epoch=None, lr_decay=0.0, 
		callbacks=[], return_history=False, verbose=False, n_jobs=1,
		multiple_check_input=True):
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

		weights : array-like or None, optional
			An array of weights, one for each sequence to train on. If None,
			all sequences are equally weighted. Default is None.

		labels : array-like or None, optional
			An array of state labels for each sequence. This is only used in
			'labeled' training. If used this must be comprised of n lists where
			n is the number of sequences to train on, and each of those lists
			must have one label per observation. A None in this list corresponds
			to no labels for the entire sequence and triggers semi-supervised
			learning, where the labeled sequences are summarized using labeled
			fitting and the unlabeled are summarized using the specified algorithm.
			Default is None.

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

		batches_per_epoch : int or None, optional
			The number of batches in an epoch. This is the number of batches to
			summarize before calling `from_summaries` and updating the model
			parameters. This allows one to do minibatch updates by updating the
			model parameters before setting the full dataset. If set to None,
			uses the full dataset. Default is None.

		lr_decay : double, optional, positive
			The step size decay as a function of the number of iterations.
			Functionally, this sets the inertia to be (2+k)^{-lr_decay}
			where k is the number of iterations. This causes initial
			iterations to have more of an impact than later iterations,
			and is frequently used in minibatch learning. This value is
			suggested to be between 0.5 and 1. Default is 0, meaning no
			decay.

		callbacks : list, optional
			A list of callback objects that describe functionality that should
			be undertaken over the course of training.

		return_history : bool, optional
			Whether to return the history during training as well as the model.

		verbose : bool, optional
			Whether to print the improvement in the model fitting at each
			iteration. Default is True.

		n_jobs : int, optional
			The number of threads to use when performing training. This
			leads to exact updates. Default is 1.

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
		cdef int mv = self.multivariate
		cdef double improvement = INF
		cdef double total_improvement = 0
		cdef double initial_log_probability_sum
		cdef double log_probability_sum
		cdef double last_log_probability_sum
		cdef str alg = algorithm.lower()
		cdef bint check_input = alg == 'viterbi'

		training_start_time = time.time()

		if multiple_check_input:
			# if we should check input multiple times
			# use old code, where we only change class of input
			# checking input will be in `summarize` function
			if not isinstance(sequences, BaseGenerator):
				data_generator = SequenceGenerator(sequences, weights, labels)
			else:
				data_generator = sequences
		else:
			if not isinstance(sequences, BaseGenerator):
				#check_input:
				#sequences have elements which are ndarrays. For each dimension in
				#our HMM model we have one row in this array, so we have to
				#iterate over all sequences for all dimensions
				checked_sequences = []
				for sequence in sequences:
					sequence_ndarray = _check_input(sequence, self)
					checked_sequences.append(sequence_ndarray)

				if labels is not None:
					labels = numpy.array(labels)

				data_generator = SequenceGenerator(checked_sequences, weights, 
					labels)
			else:
				checked_sequences, checked_weights, checked_labels = [], [], []

				for batch in sequences.batches():
					sequence_ndarray= _check_input(batch[0][0],self)
					checked_sequences.append(sequence_ndarray)
					checked_weights.append(batch[1])

					if len(batch) == 3:
						checked_labels.append(batch[2][0])
					else:
						checked_labels = None

				data_generator = SequenceGenerator(checked_sequences, checked_weights,
					checked_labels)

		n = data_generator.shape[0]

		semisupervised = False
		if labels is not None:
			for l in labels:
				if l is None:
					semisupervised = True
					break

		batches_per_epoch = batches_per_epoch 
		n_seen_batches = 0

		with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
			f = delayed(self.summarize)

			while improvement > stop_threshold or iteration < min_iterations + 1:
				epoch_start_time = time.time()
				step_size = None if inertia is None else 1 - ((1 - inertia) * (2 + iteration) ** -lr_decay)

				self.from_summaries(step_size, pseudocount, transition_pseudocount,
					emission_pseudocount, use_pseudocount,
					edge_inertia, distribution_inertia)

				if iteration >= max_iterations + 1:
					break

				if semisupervised:
					log_probability_sum = sum(parallel(f(*batch, algorithm='labeled', 
						check_input=multiple_check_input) for batch in data_generator.labeled_batches()))

					log_probability_sum += sum(parallel(f(*batch, algorithm=algorithm, 
						check_input=multiple_check_input) for batch in data_generator.unlabeled_batches()))


				elif labels is not None:
					log_probability_sum = sum(parallel(f(*batch, 
						algorithm=algorithm, check_input=multiple_check_input) for batch in data_generator.batches()))

				else:
					log_probability_sum = sum(parallel(f(*batch, algorithm=algorithm,
						check_input=multiple_check_input) for batch in data_generator.batches()))

				if iteration == 0:
					initial_log_probability_sum = log_probability_sum
				else:
					epoch_end_time = time.time()
					time_spent = epoch_end_time - epoch_start_time
					improvement = log_probability_sum - last_log_probability_sum

					total_improvement += improvement

				iteration += 1
				last_log_probability_sum = log_probability_sum

		self.clear_summaries()
		return self

	def summarize(self, sequences, weights=None, labels=None, algorithm='baum-welch',
		check_input=True):
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

		weights : array-like or None, optional
			An array of weights, one for each sequence to train on. If None,
			all sequences are equally weighted. Default is None.

		labels : array-like or None, optional
			An array of state labels for each sequence. This is only used in
			'labeled' training. If used this must be comprised of n lists where
			n is the number of sequences to train on, and each of those lists
			must have one label per observation. Default is None.

		algorithm : 'baum-welch', 'viterbi', 'labeled'
			The training algorithm to use. Baum-Welch uses the forward-backward
			algorithm to train using a version of structured EM. Viterbi
			iteratively runs the sequences through the Viterbi algorithm and
			then uses hard assignments of observations to states using that.
			Default is 'baum-welch'. Labeled training requires that labels
			are provided for each observation in each sequence.

		check_input : bool, optional
			Check the input. This casts the input sequences as numpy arrays,
			and converts non-numeric inputs into numeric inputs for faster
			processing later. Default is True.

		Returns
		-------
		logp : double
			The log probability of the sequences.
		"""

		cdef int mv = self.multivariate
		cdef list X = []

		if self.d == 0:
			raise ValueError("must bake model before summarizing data")

		if check_input:
			if weights is None:
				weights = numpy.ones(len(sequences), dtype='float64')
			else:
				weights = numpy.array(weights, dtype='float64')

			if labels is not None:
				labels = numpy.array(labels)

			for sequence in sequences:
				sequence_ndarray = _check_input(sequence, self)
				X.append(sequence_ndarray)
		else:
			X = sequences

		if algorithm == 'baum-welch':
			return sum([self._baum_welch_summarize(sequence, weight)
				for sequence, weight in zip(X, weights)])

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

		if self.finite == 1:
			log_sequence_probability = f[n*m + self.end_index]
		else:
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
					if self.discrete:
						self.states[k].distribution.from_summaries(
							distribution_inertia, emission_pseudocount)
					else:
						self.states[k].distribution.from_summaries(
							distribution_inertia)

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
	def from_matrix(cls, transition_probabilities, distributions, starts, ends=None,
		state_names=None, name=None, verbose=False, merge='None'):
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

		ends : array-like, shape (n_states), optional
			If passed in, the probabilities of ending in each of the states.
			If ends is None, then assumes the model has no explicit end
			state. Default is None.

		state_names : array-like, shape (n_states), optional
			The name of the states. If None is passed in, default names are
			generated. Default is None

		name : str, optional
			The name of the model. Default is None

		verbose : bool, optional
			The verbose parameter for the underlying bake method. Default is False.

		merge : 'None', 'Partial', 'All', optional
			The merge parameter for the underlying bake method. Default is All

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
		model = HiddenMarkovModel(name=name)
		state_names = state_names or ["s{}".format(i) for i in range(len(distributions))]

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

		if ends is not None:
			# Connect states to the end of the model if a non-zero probability
			for i, prob in enumerate(ends):
				if prob != 0:
					model.add_transition(states[i], model.end, prob)

		model.bake(verbose=verbose, merge=merge)
		return model

	@classmethod
	def from_samples(cls, distribution, n_components, X):
		"""Learn the transitions and emissions of a model directly from data.

		This method will learn both the transition matrix, emission distributions,
		and start probabilities for each state. This will only return a dense
		graph without any silent states or explicit transitions to an end state.
		Currently all components must be defined as the same distribution, but
		soon this restriction will be removed.

		If learning a multinomial HMM over discrete characters, the initial
		emisison probabilities are initialized randomly. If learning a
		continuous valued HMM, such as a Gaussian HMM, then kmeans clustering
		is used first to identify initial clusters.

		Regardless of the type of model, the transition matrix and start
		probabilities are initialized uniformly. Then the specified learning
		algorithm (Baum-Welch recommended) is used to refine the parameters
		of the model.

		Parameters
		----------
		distribution : callable
			The emission distribution of the components of the model.

		n_components : int
			The number of states (or components) to initialize.

		X : array-like or generator
			An array of some sort (list, numpy.ndarray, tuple..) of sequences,
			where each sequence is a numpy array, which is 1 dimensional if
			the HMM is a one dimensional array, or multidimensional if the HMM
			supports multiple dimensions. Alternatively, a data generator
			object that yields sequences.

		Returns
		-------
		model : HiddenMarkovModel
			The model fit to the data.
		"""

		data_generator = SequenceGenerator(X)
		initialization_batch_size = len(data_generator)

		X_ = []
		data = data_generator.batches()
		for i in range(initialization_batch_size):
			batch = next(data)
			X_.extend(batch[0])

		X_concat = numpy.concatenate(X_)
		if X_concat.ndim == 1:
			X_concat = X_concat.reshape(-1, 1)

		kmeans = KMeans(n_clusters=n_components, random_state=0)
		y_means = kmeans.fit_predict(X_concat)

		X_ = [X_concat[y_means == i] for i in range(n_components)]
		distributions = _initialize_distributions(X_, distribution)

		k = n_components
		transition_matrix = numpy.ones((k, k)) / k
		start_probabilities = numpy.zeros(k)
		start_probabilities[0] = 1

		model = HiddenMarkovModel.from_matrix(transition_matrix, distributions, start_probabilities)

		return model
