import numpy as np


def _estimated_params(N):
    """
    calculate the number of estimated parameters in the model - "k"
    :param N: number of hidden states
    :return: k
    """
    return N * N + 2 * N - 1


def aic_criteria(log_L, N):
    """
    calculate the Akaike information criterion

    :param log_L: logarithmised likelihood of the model
    :type log_L: float
    :param N: number of hidden states
    :type N: int
    :return: the Akaike information criterion
    :rtype: float
    """
    k = _estimated_params(N)
    return -2 * log_L + 2 * k


def bic_criteria(log_L, N, M):
    """
    calculate the Bayesian information criterion

    :param log_L: logarithmised likelihood of the model
    :type log_L: float
    :param N: number of hidden states
    :type N: int
    :param M: number of observation points
    :tyoe N: int
    :return: the Bayesian information criterion
    :rtype: float
    """
    k = _estimated_params(N)
    return -2 * log_L + k * np.log(M)


def hqc_criteria(log_L, N, M):
    """
    calculate the Hannan–Quinn information criterion

    :param log_L: logarithmised likelihood of the model
    :type log_L: float
    :param N: number of hidden states
    :type N: int
    :return: the Hannan–Quinn information criterion
    :rtype: float
    """
    k = _estimated_params(N)
    return -2 * log_L + k * np.log(np.log(M))


def caic_criteria(log_L, N, M):
    """
    calculate the Bozdogan Consistent Akaike information criterion

    :param log_L: logarithmised likelihood of the model
    :type log_L: float
    :param N: number of hidden states
    :type N: int
    :return: the Bozdogan Consistent Akaike information criterion
    :rtype: float
    """
    k = _estimated_params(N)
    return -2 * log_L + k * (np.log(M) + 1)
