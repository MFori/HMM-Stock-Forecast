from math import sqrt, isnan
import numpy

from .utils import weight_set, log

# Define some useful constants
NEGINF = float("-inf")
INF = float("inf")
SQRT_2_PI = 2.50662827463
LOG_2_PI = 1.83787706641


class NormalDistribution:
    """A normal distribution based on a mean and standard deviation."""

    def __init__(self, mean, std, frozen=False, min_std=0.0):
        self.d = 1
        self.mu = mean
        self.sigma = std
        self.name = "NormalDistribution"
        self.frozen = frozen
        self.summaries = [0, 0, 0]
        self.log_sigma_sqrt_2_pi = - log(std * SQRT_2_PI)
        self.two_sigma_squared = 1. / (2 * std ** 2)
        self.min_std = min_std

    def __get__(self):
        return [self.mu, self.sigma]

    def __set__(self, parameters):
        self.mu, self.sigma = parameters

    def __reduce__(self):
        """Serialize distribution for pickling."""
        return self.__class__, (self.mu, self.sigma, self.frozen)

    def log_probability(self, X):
        """Return the log probability of the given X under this distribution.

        Parameters
        ----------
        X : double
            The X to calculate the log probability of (overridden for
            DiscreteDistributions)

        Returns
        -------
        logp : double
            The log probability of that point under the distribution.
        """
        n = 1 if isinstance(X, (int, float)) else len(X)

        logp_array = numpy.empty(n, dtype='float64')
        X_ndarray = numpy.asarray(X, dtype='float64')

        self._log_probability(X_ndarray, logp_array, n)

        if n == 1:
            return logp_array[0]
        else:
            return logp_array

    def _log_probability(self, X, log_probability, n):
        for i in range(n):
            if isnan(X[i]):
                log_probability[i] = 0.
            else:
                log_probability[i] = self.log_sigma_sqrt_2_pi - ((X[i] - self.mu) ** 2) * \
                                     self.two_sigma_squared

    def fit(self, items, weights=None, inertia=0.0, column_idx=0):
        """
        Set the parameters of this Distribution to maximize the likelihood of
        the given sample. Items holds some sort of sequence. If weights is
        specified, it holds a sequence of value to weight each item by.
        """

        if self.frozen:
            return

        self.summarize(items, weights, column_idx)
        self.from_summaries(inertia)

    def summarize(self, items, weights=None, column_idx=0):
        """
        Take in a series of items and their weights and reduce it down to a
        summary statistic to be used in training later.
        """

        items, weights = weight_set(items, weights)
        if weights.sum() <= 0:
            return

        n = items.shape[0]
        d = 1

        if items.ndim == 2:
            d = items.shape[1]

        self._summarize(items, weights, n, column_idx, d)

    def _summarize(self, items, weights, n, column_idx, d):
        x_sum = 0.0
        x2_sum = 0.0
        w_sum = 0.0

        for i in range(n):
            item = items[i * d + column_idx]
            if isnan(item):
                continue

            w_sum += weights[i]
            x_sum += weights[i] * item
            x2_sum += weights[i] * item * item

        self.summaries[0] += w_sum
        self.summaries[1] += x_sum
        self.summaries[2] += x2_sum

    def from_summaries(self, inertia=0.0):
        """
        Takes in a series of summaries, represented as a mean, a variance, and
        a weight, and updates the underlying distribution. Notes on how to do
        this for a Gaussian distribution were taken from here:
        http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
        """

        # If no summaries stored or the summary is frozen, don't do anything.
        if self.summaries[0] < 1e-8 or self.frozen == True:
            return

        mu = self.summaries[1] / self.summaries[0]
        var = self.summaries[2] / self.summaries[0] - self.summaries[1] ** 2.0 / self.summaries[0] ** 2.0

        sigma = sqrt(var)
        if sigma < self.min_std:
            sigma = self.min_std

        self.mu = self.mu * inertia + mu * (1 - inertia)
        self.sigma = self.sigma * inertia + sigma * (1 - inertia)
        self.summaries = [0, 0, 0]
        self.log_sigma_sqrt_2_pi = -log(sigma * SQRT_2_PI)
        self.two_sigma_squared = 1. / (2 * sigma ** 2) if sigma > 0 else 0

    def clear_summaries(self):
        """Clear the summary statistics stored in the object."""

        self.summaries = [0, 0, 0]

    @classmethod
    def blank(cls):
        return cls(0, 1)
