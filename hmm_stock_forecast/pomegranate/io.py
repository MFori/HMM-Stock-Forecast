import numpy


class BaseGenerator(object):
    """The base data generator class.

    This object is inherited by data generator objects in order to specify that
    they are data generators. Do not use this object directly.
    """

    def __init__(self):
        pass

    def __len__(self):
        return NotImplementedError

    @property
    def shape(self):
        return NotImplementedError

    @property
    def classes(self):
        return NotImplementedError

    @property
    def ndim(self):
        return NotImplementedError


class SequenceGenerator(BaseGenerator):
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

    def __len__(self):
        return len(self.X)

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
        for idx in range(len(self)):
            if self.y is not None:
                yield self.X[idx:idx + 1], self.weights[idx:idx + 1], self.y[idx:idx + 1]
            else:
                yield self.X[idx:idx + 1], self.weights[idx:idx + 1]

    def labeled_batches(self):
        X = [x for x, y in zip(self.X, self.y) if y is not None]
        weights = [w for w, y in zip(self.weights, self.y) if y is not None]
        y = [y for y in self.y if y is not None]

        for idx in range(len(X)):
            yield X[idx:idx + 1], weights[idx:idx + 1], y[idx:idx + 1]

    def unlabeled_batches(self):
        X = [x for x, y in zip(self.X, self.y) if y is None]
        weights = [w for w, y in zip(self.weights, self.y) if y is None]

        for idx in range(len(X)):
            yield X[idx:idx + 1], weights[idx:idx + 1]
