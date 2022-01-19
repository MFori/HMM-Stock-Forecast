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
from scipy import special
from scipy.special import logsumexp
from sklearn import cluster
from scipy.stats import multivariate_normal

from hmm_stock_forecast.hmm.utils import (
    init_covars, fill_covars, concatenate_observation_sequences,
    normalise,
    log_mask_zero,
)


class HMM(object):
    """Base class for the Hidden Markov Models. It allows for training evaluation and sampling from the HMM. 

    :param n_states: number of hidden states in the model
    :type n_states: int, optional
    """

    def __init__(
            self,
            n_states=4,
            n_emissions=4,
            covariance_type='diagonal',
            covars_prior=1e-2,
            covars_weight=1,
            min_covar=1e-3,
    ):
        """Constructor method."""
        self.N = n_states
        self.n_emissions = n_emissions
        self.covariance_type = covariance_type
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.min_covar = min_covar

    # ----------------------------------------------------------------------- #
    #        Public methods. These are callable when using the class.         #
    # ----------------------------------------------------------------------- #
    @property
    def covars(self):
        """Return covariances as a full matrix."""
        return fill_covars(
            self._covars, self.covariance_type, self.N, self.n_emissions
        )

    # Solution to Problem 1 - compute P(O|model)
    def log_likelihood(self, obs):
        """Forward-Backward procedure is used to efficiently calculate the probability of the observations, given the model - P(O|model).

        :param obs: an observation sequence
        :type obs: array_like
        :return: the log of the probability, i.e. the log likehood model, give the 
            observation - logL(model|O).
        :rtype: float
        """
        b_map = self._map_B(obs)
        return self._log_likelihood(obs, b_map)

    # Init model params from sample data - must be called before training
    def init_params(self, sample):
        self._init_params([sample])

    # Solution to Problem 3 - adjust the model parameters to maximise P(O,model)
    def train(self, obs, n_iter=100, eps=0.1) -> None:
        """Updates the HMMs parameters given a new set of observed sequences.
        The observations can either be a single (1D) array of observed symbols, or a 2D array (matrix), where each row denotes a multivariate time sample (multiple features). The model parameters are reinitialised 'n_init' times. For each initialisation the updated model parameters and the log-likelihood is stored and the best model is selected at the end.

        :param obs: a list of arrays containing the observation
                sequences of different lengths
        :type obs: list
        :param n_iter: max number of iterations to run for each initialisation; defaults to 100
        :type n_iter: int, optional
        :param eps: the threshold for the likelihood increase (convergence); defaults to 0.1
        :type eps: float, optional
        """
        self._train(obs, n_iter, eps)

    # ----------------------------------------------------------------------- #
    #             Private methods. These are used internally only.            #
    # ----------------------------------------------------------------------- #
    # noinspection PyTypeChecker
    def _log_likelihood(self, obs, b_map) -> float:
        # alpha_t(x) = P(O1...Ot,qt=Sx|model) - The probability of state x and the
        # observation up to time t, given the model.
        alpha = self.forward(obs, b_map)
        return logsumexp(alpha[-1])

    def _init_params(self, sample):
        """Initialises model parameters prior to fitting. If init_type if random, it samples from a Dirichlet distribution according to the given priors. Otherwise it initialises the starting probabilities and transition probabilities uniformly.

        :param sample: list of observation sequences used to find the initial state means and covariances for the Gaussian and Heterogeneous models
        :type sample: list, optional
        """
        self.A = np.ones((self.N, self.N)) / self.N
        self.pi = np.zeros(self.N)
        self.pi[0] = 1

        X_concat = concatenate_observation_sequences(sample)

        kmeans = cluster.KMeans(n_clusters=self.N)
        kmeans.fit(X_concat)
        self.means = kmeans.cluster_centers_

        cv = np.cov(X_concat.T) + self.min_covar * np.eye(self.n_emissions)
        self._covars = init_covars(cv, self.covariance_type, self.N)

    def forward(self, obs_seq, B_map):
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
        alpha = np.zeros((n_samples, self.N))
        log_pi = log_mask_zero(self.pi)
        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(B_map)

        # init stage - alpha_1(i) = pi(i)b_i(o_1)
        for i in range(self.N):
            alpha[0][i] = log_pi[i] + log_B_map[i][0]

        # induction
        buffer = np.zeros(self.N)
        for t in range(1, n_samples):
            for j in range(self.N):
                for i in range(self.N):
                    buffer[i] = alpha[t - 1][i] + log_A[i][j]
                alpha[t][j] = logsumexp(buffer) + log_B_map[j][t]

        return alpha

    def backward(self, obs_seq, B_map):
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
        beta = np.zeros((n_samples, self.N))

        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(B_map)

        # init stage
        for i in range(self.N):
            beta[len(obs_seq) - 1][i] = 0.0

        # induction
        work_buffer = np.zeros(self.N)
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.N):
                for j in range(self.N):
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
            alpha = self.forward(obs_seq, B_map)
        if beta is None:
            beta = self.backward(obs_seq, B_map)

        n_samples = len(obs_seq)

        # The xi variable is a np array indexed by time, state, and state (TxNxN).
        # xi[t][i][j] = the probability of being in state 'i' at time 't', and 'j' at
        # time 't+1' given the entire observation sequence.
        log_xi_sum = np.zeros((self.N, self.N))
        work_buffer = np.full((self.N, self.N), -np.inf)

        # compute the logarithm of the parameters
        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(B_map)
        logprob = logsumexp(alpha[n_samples - 1])

        for t in range(n_samples - 1):
            for i in range(self.N):
                for j in range(self.N):
                    work_buffer[i, j] = (
                            alpha[t][i]
                            + log_A[i][j]
                            + log_B_map[j][t + 1]
                            + beta[t + 1][j]
                            - logprob
                    )

            for i in range(self.N):
                for j in range(self.N):
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

        with np.errstate(under='ignore'):
            a_lse = special.logsumexp(log_gamma, 1, keepdims=True)
        log_gamma -= a_lse

        with np.errstate(under='ignore'):
            return np.exp(log_gamma)

    # Methods used by self.train()
    def _train(self, obs, n_iter=100, eps=0.1) -> None:
        """Training is repeated 'n_iter' times, or until log-likelihood of the model increases by less than a threshold.
        """
        old_log_likelihood = np.nan
        for it in range(n_iter):
            B_map = self._map_B(obs)
            # calculate the log likelihood of the previous model
            # we compute the P(O|model) for the set of old parameters
            log_likelihood = self._log_likelihood(obs, B_map)

            stats = self._compute_intermediate_values(obs, B_map)

            # perform the M-step to update the model parameters
            new_model = self._M_step(stats)

            gamma = stats['gamma']
            self._update_means_covars(gamma, obs)

            self.pi = gamma[0]
            normalise(self.pi)
            self.A = new_model['A']

            improvement = abs(log_likelihood - old_log_likelihood) / abs(old_log_likelihood)
            old_log_likelihood = log_likelihood
            if improvement <= eps:
                break

    def _compute_intermediate_values(self, obs, B_map):
        """Calculates the various intermediate values for the Baum-Welch on a list of observation sequences.

        :param obs: a list of ndarrays/lists containing
                the observation sequences. Each sequence can be the same or of
                different lengths.
        :type obs: list
        :return: a dictionary of sufficient statistics required for the M-step
        :rtype: dict
        """
        stats = {
            'A': np.zeros((self.N, self.N)),
        }

        # calculate the log likelihood of the previous model
        # we compute the P(O|model) for the set of old parameters
        log_likelihood = self._log_likelihood(obs, B_map)

        alpha = self.forward(obs, B_map)
        beta = self.backward(obs, B_map)

        obs_stats = {}

        obs_stats['xi'] = self._calc_xi(
            obs,
            B_map=B_map,
            alpha=alpha,
            beta=beta,
        )
        stats['gamma'] = self._calc_gamma(alpha, beta)

        with np.errstate(under='ignore'):
            stats['A'] += np.exp(obs_stats['xi'])

        return stats

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

        A_ = np.maximum(stats['A'], 0)
        new_model['A'] = np.where(self.A == 0, 0, A_)
        normalise(new_model['A'], axis=1)

        return new_model

    def _update_means_covars(self, gamma, obs):
        gamma_sum = np.zeros_like(self.means)
        gamma_obs_sum = np.zeros_like(self.means)

        for i in range(self.N):
            for t, o in enumerate(obs[:-1]):
                gamma_sum += gamma[t][i]
                gamma_obs_sum += gamma[i][i] * o

        means = gamma_obs_sum / gamma_sum
        self.means = means

        gamma_sum = np.zeros_like(self.means)
        gamma_obs_sum = np.zeros_like(self.means)
        for i in range(self.N):
            for t, o in enumerate(obs):
                gamma_sum += gamma[t][i]
                gamma_obs_sum += gamma[t][i] * (o - means[i])

        covars = gamma_obs_sum / gamma_sum
        self._covars = np.abs(covars)  # todo why covars are negative??

    def _map_B(self, obs_seq):
        """Deriving classes should implement this method, so that it maps the
        observations' mass/density Bj(Ot) to Bj(t). The purpose of this method is to create a common parameter that will conform both to the discrete case where PMFs are used, and the continuous case where PDFs are used.

        :param obs_seq: an observation sequence of shape (n_samples, n_features)
        :type obs_seq: array_like
        :return: the mass/density mapping of shape (n_states, n_samples)
        :rtype: array_like
        """
        B_map = np.zeros((self.N, len(obs_seq)))

        for j in range(self.N):
            for t in range(len(obs_seq)):
                B_map[j][t] = self._pdf(obs_seq[t], self.means[j], self.covars[j])

        return B_map

    def _pdf(self, x, mean, covar):
        """Multivariate Gaussian PDF function.

        :param x: a multivariate sample
        :type x: array_like
        :param mean: mean of the distribution
        :type mean: array_like
        :param covar: covariance matrix of the distribution
        :type covar: array_like
        :return: the PDF of the sample
        :rtype: float
        """
        if not np.all(np.linalg.eigvals(covar) > 0):
            covar = covar + self.min_covar * np.eye(self.n_emissions)
        return multivariate_normal.pdf(x, mean=mean, cov=covar, allow_singular=True)
