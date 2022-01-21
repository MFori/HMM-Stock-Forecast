import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

from hmm_stock_forecast.hmm.ihmm import IHMM
from hmm_stock_forecast.hmm.utils import (normalise, log_mask_zero)

MIN_COVAR = 1e-3


class HMM(IHMM):
    N: int  # number of hidden states
    n_emissions: int  # number of features in each state
    pi: np.array  # start probabilities
    A: np.array  # transition probability matrix
    means: np.array  # gaussian means
    covars: np.array  # gaussian covariances
    b_map: np.array

    def __init__(self, n_states=4, n_emissions=1):
        self.N = n_states
        self.n_emissions = n_emissions

    # same naming as pomegranate
    # just return self.log_likelihood
    def log_probability(self, obs) -> float:
        return self.log_likelihood(obs)

    # noinspection PyTypeChecker
    def log_likelihood(self, obs, b_map=None) -> float:
        """Forward-Backward procedure is used to efficiently calculate the probability of the observations, given the model - P(O|model).

        :param b_map:
        :param obs: an observation sequence
        :type obs: array_like
        :return: the log of the probability, i.e. the log likehood model, give the
            observation - logL(model|O).
        :rtype: float
        """
        if b_map is None:
            b_map = self._map_B(obs)

        alpha = self.forward(obs, b_map)
        return logsumexp(alpha[-1])

    def init_params(self, sample) -> None:
        """Initialises model parameters prior to fitting. If init_type if random, it samples from a Dirichlet distribution according to the given priors. Otherwise it initialises the starting probabilities and transition probabilities uniformly.

        :param sample: list of observation sequences used to find the initial state means and covariances for the Gaussian and Heterogeneous models
        :type sample: list, optional
        """
        self.A = np.ones((self.N, self.N)) / self.N
        self.pi = np.zeros(self.N)
        self.pi[0] = 1

        data_for_clustering = []
        for item in sample:
            data_for_clustering.append(item)
        data_for_clustering = np.asarray(data_for_clustering, dtype=float)

        kmeans = KMeans(n_clusters=self.N)
        kmeans.fit(data_for_clustering)
        self.means = kmeans.cluster_centers_

        cv = np.cov(data_for_clustering.T) + MIN_COVAR * np.eye(self.n_emissions)
        cv = np.tile(np.diag(cv), (self.N, 1))
        self.covars = cv

    # same naming as pomegranate
    # just call self.train
    def fit(self, obs, n_iter=100, eps=0.1) -> None:
        self.train(obs, n_iter, eps)

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
        old_log_likelihood = np.nan
        for it in range(n_iter):
            self.b_map = self._map_B(obs)

            alpha = self.forward(obs, self.b_map)
            beta = self.backward(obs)
            xi = self._calc_xi(obs, alpha, beta)
            gamma = self._calc_gamma(alpha, beta)
            means = self._calc_means(obs, gamma)
            covars = self._calc_covars(obs, gamma, means)

            pi = gamma[0]
            normalise(pi)

            A = np.zeros((self.N, self.N))
            with np.errstate(under='ignore'):
                A += np.exp(xi)

            A_ = np.maximum(A, 0)
            A = np.where(self.A == 0, 0, A_)
            normalise(A, axis=1)

            self.pi = pi
            self.A = A
            self.means = means
            self.covars = covars

            # we compute the P(O|model) for the set of new parameters
            log_likelihood = self.log_likelihood(obs, self.b_map)

            improvement = abs(log_likelihood - old_log_likelihood) / abs(old_log_likelihood)
            old_log_likelihood = log_likelihood
            if improvement <= eps:
                break

    def forward(self, obs_seq, b_map):
        """Calculates 'alpha' the forward variable given an observation sequence.

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
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
        log_B_map = log_mask_zero(b_map)

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

    def backward(self, obs_seq):
        """Calculates 'beta', the backward variable for each observation sequence.

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :return: array of shape (n_samples, n_states) containing the backward variables
        :rtype: array_like
        """
        n_samples = len(obs_seq)

        # The beta variable is a ndarray indexed by time, then state (TxN).
        # beta[t][i] = the probability of being in state 'i' and then observing the
        # symbols from t+1 to the end (T).
        beta = np.zeros((n_samples, self.N))

        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(self.b_map)

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

    def _calc_xi(self, obs_seq, alpha=None, beta=None):
        """Calculates 'xi', a joint probability from the 'alpha' and 'beta' variables.

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :param alpha: array of shape (n_samples, n_states) containing the forward variables
        :type alpha: array_like, optional
        :param beta: array of shape (n_samples, n_states) containing the backward variables
        :type beta: array_like, optional
        :return: array of shape (n_samples, n_states, n_states) containing the a joint probability from the 'alpha' and 'beta' variables
        :rtype: array_like
        """
        n_samples = len(obs_seq)

        # The xi variable is a np array indexed by time, state, and state (TxNxN).
        # xi[t][i][j] = the probability of being in state 'i' at time 't', and 'j' at
        # time 't+1' given the entire observation sequence.
        log_xi_sum = np.zeros((self.N, self.N))
        work_buffer = np.full((self.N, self.N), -np.inf)

        # compute the logarithm of the parameters
        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(self.b_map)
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
            a_lse = logsumexp(log_gamma, 1, keepdims=True)
        log_gamma -= a_lse

        with np.errstate(under='ignore'):
            return np.exp(log_gamma)

    def _calc_means(self, obs, gamma):
        """
        """
        gamma_sum = np.zeros_like(self.means)
        gamma_obs_sum = np.zeros_like(self.means)

        for i in range(self.N):
            for t, o in enumerate(obs[:-1]):
                gamma_sum += gamma[t][i]
                gamma_obs_sum += gamma[i][i] * o

        return gamma_obs_sum / gamma_sum

    def _calc_covars(self, obs, gamma, means):
        """
        """
        gamma_sum = np.zeros_like(self.means)
        gamma_obs_sum = np.zeros_like(self.means)
        for i in range(self.N):
            for t, o in enumerate(obs):
                gamma_sum += gamma[t][i]
                gamma_obs_sum += gamma[t][i] * (o - means[i]) * (o - means[i])

        return np.abs(gamma_obs_sum / gamma_sum)  # todo why covars are negative??

    def _map_B(self, obs_seq):
        """Deriving classes should implement this method, so that it maps the
        observations' mass/density Bj(Ot) to Bj(t). The purpose of this method is to create a common parameter that will conform both to the discrete case where PMFs are used, and the continuous case where PDFs are used.

        :param obs_seq: an observation sequence of shape (n_samples, n_features)
        :type obs_seq: array_like
        :return: the mass/density mapping of shape (n_states, n_samples)
        :rtype: array_like
        """
        b_map = np.zeros((self.N, len(obs_seq)))

        new_covars = np.array(self.covars, copy=True)
        covars = np.array(list(map(np.diag, new_covars)))

        for j in range(self.N):
            for t in range(len(obs_seq)):
                b_map[j][t] = self._pdf(obs_seq[t], self.means[j], covars[j])

        return b_map

    def _pdf(self, sample, mean, covar):
        """Multivariate Gaussian PDF function.

        :param sample: a multivariate sample
        :type sample: array_like
        :param mean: mean of the distribution
        :type mean: array_like
        :param covar: covariance matrix of the distribution
        :type covar: array_like
        :return: the PDF of the sample
        :rtype: float
        """
        if not np.all(np.linalg.eigvals(covar) > 0):
            covar = covar + MIN_COVAR * np.eye(self.n_emissions)
        return multivariate_normal.pdf(sample, mean=mean, cov=covar, allow_singular=True)
