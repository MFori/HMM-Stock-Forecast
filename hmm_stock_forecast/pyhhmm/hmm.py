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

from .utils import (
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
        self.n_states = n_states
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
            self._covars, self.covariance_type, self.n_states, self.n_emissions
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

    def _log_likelihood(self, obs, b_map):
        # alpha_t(x) = P(O1...Ot,qt=Sx|model) - The probability of state x and the
        # observation up to time t, given the model.
        alpha = self.forward(obs, b_map)
        return logsumexp(alpha[-1])

    # Solution to Problem 3 - adjust the model parameters to maximise P(O,model)
    def train(
            self,
            obs_sequences,
            n_iter=100,
            conv_thresh=0.1,
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
             :return: the updated model
        :rtype: object
        :return: the log_likelihood of the best model
        :rtype: float
        """

        logL = self._train(
            obs_sequences,
            n_iter=n_iter,
            conv_thresh=conv_thresh,
        )

        return self, logL

    # ----------------------------------------------------------------------- #
    #             Private methods. These are used internally only.            #
    # ----------------------------------------------------------------------- #
    def _init_model_params(self, X):
        """Initialises model parameters prior to fitting. If init_type if random, it samples from a Dirichlet distribution according to the given priors. Otherwise it initialises the starting probabilities and transition probabilities uniformly.

        :param X: list of observation sequences used to find the initial state means and covariances for the Gaussian and Heterogeneous models
        :type X: list, optional
        """
        self.pi = np.random.dirichlet(
            alpha=np.ones(self.n_states), size=1
        )[0]

        self.A = np.random.dirichlet(
            alpha=np.ones(self.n_states), size=self.n_states
        )

        X_concat = concatenate_observation_sequences(X)

        kmeans = cluster.KMeans(n_clusters=self.n_states)
        kmeans.fit(X_concat)
        self.means = kmeans.cluster_centers_

        cv = np.cov(X_concat.T) + self.min_covar * np.eye(self.n_emissions)
        self._covars = init_covars(cv, self.covariance_type, self.n_states)

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
        alpha = np.zeros((n_samples, self.n_states))
        log_pi = log_mask_zero(self.pi)
        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(B_map)

        # init stage - alpha_1(i) = pi(i)b_i(o_1)
        for i in range(self.n_states):
            alpha[0][i] = log_pi[i] + log_B_map[i][0]

        # induction
        buffer = np.zeros(self.n_states)
        for t in range(1, n_samples):
            for j in range(self.n_states):
                for i in range(self.n_states):
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
            alpha = self.forward(obs_seq, B_map)
        if beta is None:
            beta = self.backward(obs_seq, B_map)

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

        with np.errstate(under='ignore'):
            a_lse = special.logsumexp(log_gamma, 1, keepdims=True)
        log_gamma -= a_lse

        with np.errstate(under='ignore'):
            return np.exp(log_gamma)

    # Methods used by self.train()
    def _train(
            self,
            obs_sequences,
            n_iter=100,
            conv_thresh=0.1,
    ):
        """Training is repeated 'n_iter' times, or until log-likelihood of the model increases by less than a threshold.

        :param obs_sequences: a list of arrays containing the observation
                sequences of different lengths
        :type obs_sequences: list
        :param n_iter: max number of iterations to run for each initialisation; defaults to 100
        :type n_iter: int, optional
        :param conv_thresh: the threshold for the likelihood increase (convergence); defaults to 0.1
        :type conv_thresh: float, optional
       :return: dictionary containing the updated model parameters
        :rtype: dict
        :return: the accumulated log-likelihood for all the observations. (if  return_log_likelihoods is True then the list of log-likelihood values from each iteration)
        :rtype: float

        """
        self._init_model_params(X=obs_sequences)

        curr_log_likelihood = 0
        log_likelihood_iter = []
        old_log_likelihood = np.nan
        for it in range(n_iter):

            stats, curr_log_likelihood = self._compute_intermediate_values(
                obs_sequences
            )

            # perform the M-step to update the model parameters
            new_model = self._M_step(stats)

            self.pi = new_model['pi']
            self.A = new_model['A']
            self.means = new_model['means']
            self._covars = new_model['covars']

            improvement = abs(curr_log_likelihood - old_log_likelihood) / abs(old_log_likelihood)
            if improvement <= conv_thresh:
                break

            log_likelihood_iter.append(curr_log_likelihood)
            old_log_likelihood = curr_log_likelihood

        return curr_log_likelihood

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
            log_likelihood = self._log_likelihood(obs_seq, B_map)
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
            'alpha': self.forward(obs_seq, B_map),
            'beta': self.backward(obs_seq, B_map),
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

        pi_ = np.maximum(stats['pi'], 0)
        new_model['pi'] = np.where(self.pi == 0, 0, pi_)
        normalise(new_model['pi'])

        A_ = np.maximum(stats['A'], 0)
        new_model['A'] = np.where(self.A == 0, 0, A_)
        normalise(new_model['A'], axis=1)

        denom = stats['post'][:, np.newaxis]
        new_model['means'] = stats['obs'] / denom

        if self.covariance_type in ('spherical', 'diagonal'):
            cv_num = (
                    stats['obs**2']
                    - 2 * new_model['means'] * stats['obs']
                    + new_model['means'] ** 2 * denom
            )
            cv_den = np.amax(self.covars_weight - 1, 0) + denom
            covars_new = (self.covars_prior + cv_num) / \
                         np.maximum(cv_den, 1e-5)
            if self.covariance_type == 'spherical':
                covars_new = covars_new.mean(1)
        elif self.covariance_type in ('tied', 'full'):
            cv_num = np.empty(
                (self.n_states, self.n_emissions, self.n_emissions))
            for c in range(self.n_states):
                obs_mean = np.outer(stats['obs'][c], new_model['means'][c])
                cv_num[c] = (
                        stats['obs*obs.T'][c]
                        - obs_mean
                        - obs_mean.T
                        + np.outer(new_model['means'][c],
                                   new_model['means'][c])
                        * stats['post'][c]
                )
            cvweight = np.amax(self.covars_weight - self.n_emissions, 0)
            if self.covariance_type == 'tied':
                covars_new = (self.covars_prior + cv_num.sum(axis=0)) / (
                        cvweight + stats['post'].sum()
                )
            elif self.covariance_type == 'full':
                covars_new = (self.covars_prior + cv_num) / (
                        cvweight + stats['post'][:, None, None]
                )
        new_model['covars'] = covars_new

        return new_model

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

        stats['post'] = np.zeros(self.n_states)
        stats['obs'] = np.zeros((self.n_states, self.n_emissions))
        stats['obs**2'] = np.zeros((self.n_states, self.n_emissions))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros(
                (self.n_states, self.n_emissions, self.n_emissions)
            )

        return stats

    def _accumulate_sufficient_statistics(
            self, stats, obs_stats, obs_seq
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

        stats['post'] += obs_stats['gamma'].sum(axis=0)
        stats['obs'] += self._reestimate_stat_obs(
            obs_stats['gamma'], obs_seq
        )

        if self.covariance_type in ('spherical', 'diagonal'):
            stats['obs**2'] += self._reestimate_stat_obs2(
                obs_stats['gamma'], obs_seq
            )
        elif self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] += self._reestimate_stat_obs2(
                obs_stats['gamma'], obs_seq
            )

    def _reestimate_stat_obs(self, gamma, obs_seq):
        """Helper method for the statistics accumulation. Computes the sum of
        the posteriors times the observations for the update of the means.

        :param gamma: array of shape (n_samples, n_states), the posteriors
        :type gamma: array_like
        :param obs_seq: an observation sequence
        :type obs_seq: array_like
        """
        stat_obs = np.zeros_like(self.means)

        for j in range(self.n_states):
            for t, obs in enumerate(obs_seq):
                stat_obs[j] += gamma[t][j] * obs

        return stat_obs

    def _reestimate_stat_obs2(self, gamma, obs_seq):
        """Helper method for the statistics accumulation. Computes the sum of
        the posteriors times the square of the observations for the update
        of the covariances.

        :param gamma: array of shape (n_samples, n_states), the posteriors
        :type gamma: array_like
        :param obs_seq: an observation sequence
        :type obs_seq: array_like
        """
        if self.covariance_type in ('tied', 'full'):
            stat_obs2 = np.zeros(
                (self.n_states, self.n_emissions, self.n_emissions))
        else:
            stat_obs2 = np.zeros((self.n_states, self.n_emissions))

        for j in range(self.n_states):
            for t, obs in enumerate(obs_seq):
                if self.covariance_type in ('tied', 'full'):
                    stat_obs2[j] += gamma[t][j] * np.outer(obs, obs)
                else:
                    stat_obs2[j] += np.dot(gamma[t][j], obs ** 2)
        return stat_obs2

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

    def _map_B(self, obs_seq):
        """Deriving classes should implement this method, so that it maps the
        observations' mass/density Bj(Ot) to Bj(t). The purpose of this method is to create a common parameter that will conform both to the discrete case where PMFs are used, and the continuous case where PDFs are used.

        :param obs_seq: an observation sequence of shape (n_samples, n_features)
        :type obs_seq: array_like
        :return: the mass/density mapping of shape (n_states, n_samples)
        :rtype: array_like
        """
        B_map = np.zeros((self.n_states, len(obs_seq)))

        for j in range(self.n_states):
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
