""""""
"""
Created on Nov 20, 2019
@author: semese

Parts of the code come from: https://github.com/hmmlearn/hmmlearn
"""


import numpy as np
from scipy import linalg, special

COVARIANCE_TYPES = frozenset(('spherical', 'tied', 'diagonal', 'full'))

# ---------------------------------------------------------------------------- #
#                            Utils for the HMM models                          #
# ---------------------------------------------------------------------------- #


def normalise(a, axis=None):
    """
    Normalise the input array so that it sums to 1.  Modifies the input **inplace**.

    :param a: non-normalised input data
    :type a: array_like
    :param axis: dimension along which normalisation is performed, defaults to None
    :type axis: int, optional
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum


def log_normalise(a, axis=None):
    """
    Normalise the input array so that ``sum(exp(a)) == 1``. Modifies the input **inplace**.

    :param a: non-normalised input data
    :type a: array_like
    :param axis: dimension along which normalisation is performed, defaults to None
    :type axis: int, optional
    """
    with np.errstate(under='ignore'):
        a_lse = special.logsumexp(a, axis, keepdims=True)
    a -= a_lse


def log_mask_zero(a):
    """
    Compute the log of input probabilities masking divide by zero in log.
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalised to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.
    This function masks this unharmful warning.

    :param a: input data
    :type a: array_like
    """
    a = np.asarray(a)
    with np.errstate(divide='ignore'):
        return np.log(a)


def concatenate_observation_sequences(observation_sequences, gidx=None):
    """
    Function to concatenate the observation sequences and remove the
    partially or completely missing observations to create a proper
    input for the KMeans.

    :param observation_sequences: each element is an array of observations
    :type observation_sequences: list
    :param gidx: if provided, only the specified columns will be concatenated, 
        defaults to None
    :type gidx: array_like, optional
    :return: concatenated observations without missing values
    :rtype: list
    """
    concatenated = []
    for obs_seq in observation_sequences:
        for obs in obs_seq:
            if gidx is not None:
                gidx = int(gidx)
                if not np.any(obs[:gidx] is np.nan or obs[:gidx] != obs[:gidx]):
                    concatenated.append(obs[:gidx])
            else:
                if not np.any(obs is np.nan or obs != obs):
                    concatenated.append(obs)
    return np.asarray(concatenated, dtype=float)


def init_covars(tied_cv, covariance_type, n_states):
    """
    Method for initialising the covariances based on the
    covariance type. See GaussianHMM class definition for details.

    :param tied_cv: the tied covariance matrix 
    :type tied_cv: array_like
    :param covariance_type: covariance_type: string describing the type of
            covariance parameters to use
    :type covariance_type: string
    :param n_states: number of hidden states 
    :type n_states: int
    :return: the initialised covariance matrix
    :rtype: array_like
    """
    if covariance_type == 'spherical':
        cv = tied_cv.mean() * np.ones((n_states,))
    elif covariance_type == 'tied':
        cv = tied_cv
    elif covariance_type == 'diagonal':
        cv = np.tile(np.diag(tied_cv), (n_states, 1))
    elif covariance_type == 'full':
        cv = np.tile(tied_cv, (n_states, 1, 1))
    return cv


def fill_covars(covars, covariance_type, n_states, n_features):
    """
    Return the covariance matrices in full form: (n_states, n_features, n_features)

    :param covars: the reduced form of the covariance matrix
    :type covars: array_like
    :param covariance_type: covariance_type: string describing the type of
            covariance parameters to use
    :type covariance_type: string
    :param n_states: number of hidden states 
    :type n_states: int
    :param n_features: the number of features
    :type n_features: int
    :return: the covariance matrices in full form: (n_states, n_features, n_features)
    :rtype: array_like
    """
    new_covars = np.array(covars, copy=True)
    if covariance_type == 'full':
        return new_covars
    elif covariance_type == 'diagonal':
        return np.array(list(map(np.diag, new_covars)))
    elif covariance_type == 'tied':
        return np.tile(new_covars, (n_states, 1, 1))
    elif covariance_type == 'spherical':
        eye = np.eye(n_features)[np.newaxis, :, :]
        new_covars = new_covars[:, np.newaxis, np.newaxis]
        temp = eye * new_covars
        return temp

# ---------------------------------------------------------------------------- #
#                           Model order selection utils                        #
# ---------------------------------------------------------------------------- #


def aic_hmm(log_likelihood, dof):
    """
    Function to compute the Aikaike's information criterion for an HMM given
    the log-likelihood of observations.

    :param log_likelihood: logarithmised likelihood of the model
        dof (int) - single numeric value representing the number of trainable
        parameters of the model
    :type log_likelihood: float
    :param dof: single numeric value representing the number of trainable
        parameters of the model
    :type dof: int
    :return: the Aikaike's information criterion
    :rtype: float
    """
    return -2 * log_likelihood + 2 * dof


def bic_hmm(log_likelihood, dof, n_samples):
    """
    Function to compute Bayesian information criterion for an HMM given a
    the log-likelihood of observations.

    :param log_likelihood: logarithmised likelihood of the model
        dof (int) - single numeric value representing the number of trainable
        parameters of the model
    :type log_likelihood: float
    :param dof: single numeric value representing the number of trainable
        parameters of the model
    :type dof: int
    :param n_samples: length of the time-series of observations
    :type n_samples: int
    :return: the Bayesian information criterion
    :rtype: float
    """
    return -2 * log_likelihood + dof * np.log(n_samples)
