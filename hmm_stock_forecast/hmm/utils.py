import numpy as np


def normalise(a, axis=None) -> np.array:
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


def log_mask_zero(a) -> np.array:
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
