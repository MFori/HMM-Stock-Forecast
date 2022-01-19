"""
Created on Nov 20, 2019
@authors: semese, fmorenopino

This code is based on:
 - HMM implementation by guyz- https://github.com/guyz/HMM
 - HMM implementation by fmorenopino - https://github.com/fmorenopino/HMM_eb2
 - HMM implementation by anntzer - https://github.com/hmmlearn/
 
For theoretical bases see:
 - L. R. Rabiner, A tutorial on hidden Markov models and selected applications
   in speech recognition, in Proceedings of the IEEE, vol. 77, no. 2,
   pp. 257-286, Feb. 1989.
 - K.P. Murphy, Machine Learning: A Probabilistic Perspective, The MIT Press
   ©2012, ISBN:0262018020 9780262018029
"""

import numpy as np
from scipy.special import logsumexp

from .utils import (
    log_normalise,
    normalise,
    log_mask_zero,
)


class BaseHMM(object):
    """Base class for the Hidden Markov Models. It allows for training evaluation and sampling from the HMM. 

    :param n_states: number of hidden states in the model
    :type n_states: int, optional
    :param pi_prior: array of shape (n_states, ) setting the parameters of the
        Dirichlet prior distribution for the starting probabilities. Defaults to 1.
    :type pi_prior: array_like, optional 
    :param A_prior: array of shape (n_states, ), giving the parameters of the Dirichlet 
        prior distribution for each row of the transition probabilities 'A'. Defaults to 1.
    :param pi: array of shape (n_states, ) giving the initial state
        occupation distribution 'pi'
    :type pi: array_like 
    :type A_prior: array_like, optional 
    :param A: array of shape (n_states, n_states) giving the matrix of
            transition probabilities between states
    :type A: array_like
    :param learning_rate: a value from [0,1), controlling how much
        the past values of the model parameters count when computing the new
        model parameters during training; defaults to 0.
    :type learning_rate: float, optional
    """

    def __init__(
            self,
            n_states,
            pi_prior=1.0,
            A_prior=1.0,
            learning_rate=0.,
    ):
        """Constructor method."""
        self.n_states = n_states
        self.pi_prior = pi_prior
        self.A_prior = A_prior
        self.learning_rate = learning_rate

    # ----------------------------------------------------------------------- #
    #        Public methods. These are callable when using the class.         #
    # ----------------------------------------------------------------------- #
    # Solution to Problem 1 - compute P(O|model)
    def forward(self, obs_seq, B_map=None):
        """Forward-Backward procedure is used to efficiently calculate the probability of the observations, given the model - P(O|model).

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :param B_map: mapping of the observations' mass/density Bj(Ot) to Bj(t)
        :type B_map: array_like, optional
        :return: the log of the probability, i.e. the log likehood model, give the 
            observation - logL(model|O).
        :rtype: float
        """
        if B_map is None:
            # if the emission probabilies not given, compute
            B_map = self._map_B(obs_seq)

        # alpha_t(x) = P(O1...Ot,qt=Sx|model) - The probability of state x and the
        # observation up to time t, given the model.
        alpha = self._calc_alpha(obs_seq, B_map)

        return logsumexp(alpha[-1])

    # Solution to Problem 3 - adjust the model parameters to maximise P(O,model)
    def train(
            self,
            obs_sequences,
            n_iter=100,
            conv_thresh=0.1,
            conv_iter=5,
    ):
        """Updates the HMMs parameters given a new set of observed sequences.
        The observations can either be a single (1D) array of observed symbols, or a 2D array (matrix), where each row denotes a multivariate time sample (multiple features). The model parameters are reinitialised 'n_init' times. For each initialisation the updated model parameters and the log-likelihood is stored and the best model is selected at the end.

        :param obs_sequences: a list of arrays containing the observation
                sequences of different lengths
        :type obs_sequences: list
        :param n_iter: max number of iterations to run for each initialisation; defaults to 100
        :type n_iter: int, optional
        :param conv_thresh: the threshold for the likelihood increase (convergence); defaults to 0.1
        :type conv_thresh: float, optional
        :param conv_iter: number of iterations for which the convergence criteria has to hold before 
            early-stopping; defaults to 5
        :type conv_iter: int, optional
          :return: the updated model
        :rtype: object
        :return: the log_likelihood of the best model
        :rtype: float
        """

        n_model, logL = self._train(
            obs_sequences,
            n_iter=n_iter,
            conv_thresh=conv_thresh,
            conv_iter=conv_iter,
        )

        self._update_model(n_model)
        return self, logL

    # ----------------------------------------------------------------------- #
    #             Private methods. These are used internally only.            #
    # ----------------------------------------------------------------------- #
    def _init_model_params(self):
        """Initialises model parameters prior to fitting. If init_type if random, it samples from a Dirichlet distribution according to the given priors. Otherwise it initialises the starting probabilities and transition probabilities uniformly.

        :param X: list of observation sequences used to find the initial state means and covariances for the Gaussian and Heterogeneous models
        :type X: list, optional
        """
        self.pi = np.random.dirichlet(
            alpha=self.pi_prior * np.ones(self.n_states), size=1
        )[0]

        self.A = np.random.dirichlet(
            alpha=self.A_prior * np.ones(self.n_states), size=self.n_states
        )

    def _calc_alpha(self, obs_seq, B_map):
        """Calculates 'alpha' the forward variable given an observation sequence.

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :param B_map: mapping of the observations' mass/density Bj(Ot) to Bj(t)
        :type B_map: array_like, optional
        :return: array of shape (n_samples, n_states) containing the forward variables
        :rtype: array_like
        """
        n_samples = len(obs_seq)

        # The alpha variable is a np array indexed by time, then state (TxN).
        # alpha[t][i] = the probability of being in state 'i' after observing the
        # first t symbols.
        alpha = np.zeros((n_samples, self.n_states))
        log_pi = log_mask_zero(self.pi)
        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(B_map)

        # init stage - alpha_1(i) = pi(i)b_i(o_1)
        for i in range(self.n_states):
            alpha[0][i] = log_pi[i] + log_B_map[i][0]

        # induction
        work_buffer = np.zeros(self.n_states)
        for t in range(1, n_samples):
            for j in range(self.n_states):
                for i in range(self.n_states):
                    work_buffer[i] = alpha[t - 1][i] + log_A[i][j]
                alpha[t][j] = logsumexp(work_buffer) + log_B_map[j][t]

        return alpha

    def _calc_beta(self, obs_seq, B_map):
        """Calculates 'beta', the backward variable for each observation sequence.

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :param B_map: mapping of the observations' mass/density Bj(Ot) to Bj(t)
        :type B_map: array_like, optional
        :return: array of shape (n_samples, n_states) containing the backward variables
        :rtype: array_like
        """
        n_samples = len(obs_seq)

        # The beta variable is a ndarray indexed by time, then state (TxN).
        # beta[t][i] = the probability of being in state 'i' and then observing the
        # symbols from t+1 to the end (T).
        beta = np.zeros((n_samples, self.n_states))

        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(B_map)

        # init stage
        for i in range(self.n_states):
            beta[len(obs_seq) - 1][i] = 0.0

        # induction
        work_buffer = np.zeros(self.n_states)
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    work_buffer[j] = log_A[i][j] + \
                                     log_B_map[j][t + 1] + beta[t + 1][j]
                    beta[t][i] = logsumexp(work_buffer)

        return beta

    def _calc_xi(
            self, obs_seq, B_map=None, alpha=None, beta=None
    ):
        """Calculates 'xi', a joint probability from the 'alpha' and 'beta' variables.

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :param B_map: mapping of the observations' mass/density Bj(Ot) to Bj(t)
        :type B_map: array_like, optional
        :param alpha: array of shape (n_samples, n_states) containing the forward variables
        :type alpha: array_like, optional
        :param beta: array of shape (n_samples, n_states) containing the backward variables
        :type beta: array_like, optional
        :return: array of shape (n_samples, n_states, n_states) containing the a joint probability from the 'alpha' and 'beta' variables
        :rtype: array_like
        """
        if B_map is None:
            B_map = self._map_B(obs_seq)
        if alpha is None:
            alpha = self._calc_alpha(obs_seq, B_map)
        if beta is None:
            beta = self._calc_beta(obs_seq, B_map)

        n_samples = len(obs_seq)

        # The xi variable is a np array indexed by time, state, and state (TxNxN).
        # xi[t][i][j] = the probability of being in state 'i' at time 't', and 'j' at
        # time 't+1' given the entire observation sequence.
        log_xi_sum = np.zeros((self.n_states, self.n_states))
        work_buffer = np.full((self.n_states, self.n_states), -np.inf)

        # compute the logarithm of the parameters
        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(B_map)
        logprob = logsumexp(alpha[n_samples - 1])

        for t in range(n_samples - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    work_buffer[i, j] = (
                            alpha[t][i]
                            + log_A[i][j]
                            + log_B_map[j][t + 1]
                            + beta[t + 1][j]
                            - logprob
                    )

            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi_sum[i][j] = np.logaddexp(
                        log_xi_sum[i][j], work_buffer[i][j])

        return log_xi_sum

    def _calc_gamma(self, alpha, beta):
        """Calculates 'gamma' from 'alpha' and 'beta'.

        :param alpha: array of shape (n_samples, n_states) containing the forward variables
        :type alpha: array_like
        :param beta: array of shape (n_samples, n_states) containing the backward variables
        :type beta: array_like
        :return:  array of shape (n_samples, n_states), the posteriors
        :rtype: array_like
        """
        log_gamma = alpha + beta
        log_normalise(log_gamma, axis=1)
        with np.errstate(under='ignore'):
            return np.exp(log_gamma)

    # Methods used by self.train()
    def _train(
            self,
            obs_sequences,
            n_iter=100,
            conv_thresh=0.1,
            conv_iter=5,
    ):
        """Training is repeated 'n_iter' times, or until log-likelihood of the model increases by less than a threshold.

        :param obs_sequences: a list of arrays containing the observation
                sequences of different lengths
        :type obs_sequences: list
        :param n_iter: max number of iterations to run for each initialisation; defaults to 100
        :type n_iter: int, optional
        :param conv_thresh: the threshold for the likelihood increase (convergence); defaults to 0.1
        :type conv_thresh: float, optional
        :param conv_iter: number of iterations for which the convergence criteria has to hold before early-stopping; defaults to 5
        :type conv_iter: int, optional
        :return: dictionary containing the updated model parameters
        :rtype: dict
        :return: the accumulated log-likelihood for all the observations. (if  return_log_likelihoods is True then the list of log-likelihood values from each iteration)
        :rtype: float

        """
        self._init_model_params(X=obs_sequences)

        log_likelihood_iter = []
        old_log_likelihood = np.nan
        for it in range(n_iter):

            stats, curr_log_likelihood = self._compute_intermediate_values(
                obs_sequences
            )

            # perform the M-step to update the model parameters
            new_model = self._M_step(stats)
            self._update_model(new_model)

            improvement = abs(curr_log_likelihood - old_log_likelihood) / abs(old_log_likelihood)
            if improvement <= conv_thresh:
                break

            log_likelihood_iter.append(curr_log_likelihood)
            old_log_likelihood = curr_log_likelihood

        return new_model, curr_log_likelihood

    def _compute_intermediate_values(self, obs_sequences):
        """Calculates the various intermediate values for the Baum-Welch on a list of observation sequences.

        :param obs_sequences: a list of ndarrays/lists containing
                the observation sequences. Each sequence can be the same or of
                different lengths.
        :type ob_sequences: list
        :return: a dictionary of sufficient statistics required for the M-step
        :rtype: dict
        """
        stats = self._initialise_sufficient_statistics()
        curr_log_likelihood = 0

        for obs_seq in obs_sequences:
            B_map = self._map_B(obs_seq)

            # calculate the log likelihood of the previous model
            # we compute the P(O|model) for the set of old parameters
            log_likelihood = self.forward(obs_seq, B_map)
            curr_log_likelihood += log_likelihood

            # do the E-step of the Baum-Welch algorithm
            obs_stats = self._E_step(obs_seq, B_map)

            # accumulate stats
            self._accumulate_sufficient_statistics(
                stats, obs_stats, obs_seq
            )

        return stats, curr_log_likelihood

    def _E_step(self, obs_seq, B_map):
        """Calculates required statistics of the current model, as part
        of the Baum-Welch 'E' step. Deriving classes should override (extend) 
        this method to include any additional computations their model requires.

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :param B_map: mapping of the observations' mass/density Bj(Ot) to Bj(t)
        :type B_map: array_like, optional
        :return: a dictionary containing the required statistics
        :rtype: dict
        """

        # compute the parameters for the observation
        obs_stats = {
            'alpha': self._calc_alpha(obs_seq, B_map),
            'beta': self._calc_beta(obs_seq, B_map),
        }

        obs_stats['xi'] = self._calc_xi(
            obs_seq,
            B_map=B_map,
            alpha=obs_stats['alpha'],
            beta=obs_stats['beta'],
        )
        obs_stats['gamma'] = self._calc_gamma(
            obs_stats['alpha'], obs_stats['beta']
        )
        return obs_stats

    def _M_step(self, stats):
        """Performs the 'M' step of the Baum-Welch algorithm.
        Deriving classes should override (extend) this method to include
        any additional computations their model requires.

        :param stats: dictionary containing the accumulated statistics
        :type stats: dict
        :return: a dictionary containing the updated model parameters
        :rtype: dict
        """
        new_model = {}

        pi_ = np.maximum(self.pi_prior - 1 + stats['pi'], 0)
        new_model['pi'] = np.where(self.pi == 0, 0, pi_)
        normalise(new_model['pi'])

        A_ = np.maximum(self.A_prior - 1 + stats['A'], 0)
        new_model['A'] = np.where(self.A == 0, 0, A_)
        normalise(new_model['A'], axis=1)

        return new_model

    def _update_model(self, new_model):
        """Replaces the current model parameters with the new ones.

        :param new_model: contains the new model parameters
        :type new_model: dict
        """
        self.pi = (1 - self.learning_rate) * new_model[
            'pi'
        ] + self.learning_rate * self.pi

        self.A = (1 - self.learning_rate) * \
                 new_model['A'] + self.learning_rate * self.A

    def _initialise_sufficient_statistics(self):
        """Initialises sufficient statistics required for M-step.

        :return: a dictionary having as key-value pairs { nobs - number of samples in the data; start - array of shape (n_states, ) where the i-th element corresponds to the posterior probability of the first sample being generated by the i-th state; trans (dictionary) - containing the numerator and denominator parts of the new A matrix as arrays of shape (n_states, n_states)
        :rtype: dict
        """
        stats = {
            'nobs': 0,
            'pi': np.zeros(self.n_states),
            'A': np.zeros((self.n_states, self.n_states)),
        }
        return stats

    def _accumulate_sufficient_statistics(
            self, stats, obs_stats
    ):
        """Updates sufficient statistics from a given sample.

        :param stats: dictionary containing the sufficient statistics for all observation sequences
        :type stats: dict
        :param obs_stats: dictionary containing the sufficient statistic for one 
            observation sequence
        :type stats: dict
        """
        stats['nobs'] += 1
        stats['pi'] += obs_stats['gamma'][0]
        with np.errstate(under='ignore'):
            stats['A'] += np.exp(obs_stats['xi'])

    def _sum_up_sufficient_statistics(self, stats_list):
        """Sums sufficient statistics from a given sub-set of observation sequences.

        :param stats_list: list containing the sufficient statistics from the
                different processes
        :type stats_list: list
        :return: a dictionary of sufficient statistics
        :rtype: dict
        """
        stats_all = self._initialise_sufficient_statistics()
        logL_all = 0
        for (stat_i, logL_i) in stats_list:
            logL_all += logL_i
            for stat in stat_i.keys():
                if isinstance(stat_i[stat], dict):
                    for i in range(len(stats_all[stat]['numer'])):
                        stats_all[stat]['numer'][i] += stat_i[stat]['numer'][i]
                        stats_all[stat]['denom'][i] += stat_i[stat]['denom'][i]
                else:
                    stats_all[stat] += stat_i[stat]
        return stats_all, logL_all

    # Methods that have to be implemented in the deriving classes
    def _map_B(self, obs_seq):
        """Deriving classes should implement this method, so that it maps the
        observations' mass/density Bj(Ot) to Bj(t). The purpose of this method is to create a common parameter that will conform both to the discrete case where PMFs are used, and the continuous case where PDFs are used.

        :param obs_seq: an observation sequence of shape (n_samples, n_features)
        :type obs_seq: array_like
        :return: the mass/density mapping of shape (n_states, n_samples)
        :rtype: array_like
        """
        raise NotImplementedError(
            'A mapping function for B(observable probabilities) must be implemented.'
        )
