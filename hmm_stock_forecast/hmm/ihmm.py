class IHMM:
    """
    HMM interface, to enable switching between our HMM implementation and pomegranate
    """

    def fit(self, obs):
        """
        train model
        :param obs: observation sequence
        """
        pass

    def log_probability(self, obs) -> float:
        """
        return log likelihood of observation
        :param obs: observation sequence
        :return: log likelihood
        """
        pass
