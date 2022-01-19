import uuid


class State:
    """
    Represents a state in an HMM. Holds emission distribution, but not
    transition distribution, because that's stored in the graph edges.
    """

    def __init__(self, distribution, name=None, weight=None):
        """
        Make a new State emitting from the given distribution. If distribution
        is None, this state does not emit anything. A name, if specified, will
        be the state's name when presented in output. Name may not contain
        spaces or newlines, and must be unique within a model.
        """

        self.distribution = distribution
        self.name = name or str(uuid.uuid4())
        self.weight = weight or 1.

    def __reduce__(self):
        return self.__class__, (self.distribution, self.name, self.weight)

    def tie(self, state):
        """
        Tie this state to another state by just setting the distribution of the
        other state to point to this states distribution.
        """
        state.distribution = self.distribution

    def is_silent(self):
        """
        Return True if this state is silent (distribution is None) and False
        otherwise.
        """
        return self.distribution is None
