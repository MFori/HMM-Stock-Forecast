import numpy


class SequenceGenerator(object):
    """A generator that returns batches of sequences from a data set.

    This object will wrap a data set and optionally a set of labels and will
    return sequences as requested. Due to the processing in pomegranate, only
    batches of size 1 are supported.

    Parameters
    ----------
    X : numpy.ndarray or list
        The data set to iterate over.

    weights : numpy.ndarray or list or None, optional
        The weights for each example. Default is None.

    y: numpy.ndarray or list or None, optional
        The set of labels for each example in the data set. Default is None.
    """

    def __init__(self, X, weights=None, y=None, batches_per_epoch=None):
        self.X = X
        self.y = y
        self.idx = 0

        if weights is None:
            self.weights = numpy.ones(len(X), dtype='float64')
        else:
            self.weights = weights

        if batches_per_epoch is None:
            self.batches_per_epoch = float("inf")
        else:
            self.batches_per_epoch = batches_per_epoch

    @property
    def shape(self):
        x_ = numpy.asarray(self.X[0])

        if x_.ndim == 1:
            return len(self.X), 1
        elif x_.ndim == 2:
            return len(self.X), x_.shape[1]
        else:
            raise ValueError("Data must be passed in as a list of numpy arrays.")

    @property
    def ndim(self):
        return len(self.X[0])

    @property
    def classes(self):
        if self.y is None:
            raise ValueError("No labels found for this data set.")

        return numpy.unique(self.y)

    def batches(self):
        for idx in range(len(self.X)):
            if self.y is not None:
                yield self.X[idx:idx + 1], self.weights[idx:idx + 1], self.y[idx:idx + 1]
            else:
                yield self.X[idx:idx + 1], self.weights[idx:idx + 1]
