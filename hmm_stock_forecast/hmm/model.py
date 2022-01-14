from hmmlearn.hmm import GaussianHMM


class HMMStockForecastModel:
    def __init__(self):
        self.hmm = GaussianHMM(n_components=4, covariance_type='full', algorithm='viterbi')
        pass

    def train(self, data):
        pass

    def get_predicted_data(self):
        pass

    def get_mean_error(self):
        pass

    def get_r2(self):
        pass

    def _find_optimal_states(self):
        pass

