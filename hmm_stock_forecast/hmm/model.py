from hmmlearn.hmm import GaussianHMM
import numpy as np


class HMMStockForecastModel:
    _hmm: GaussianHMM
    predicted: np.array

    def __init__(self, data, window=200):
        self.data = data
        self.window = window
        pass

    def run(self):
        size = len(self.data)
        predicted = np.empty([0, 4])
        hmm = self._hmm = GaussianHMM(n_components=4, algorithm='viterbi', init_params="stmc")

        for i in reversed(range(self.window + 1)):
            print(i)

            train = self.data[size - self.window - i:size - i, :]
            hmm.fit(train)

            likelihood = hmm.score(train)
            likelihoods = []
            j = i + 1
            while size - self.window - j > 0:
                obs = self.data[size - self.window - j:size - j, :]
                likelihoods = np.append(likelihoods, hmm.score(obs))
                j += 1

            likelihood_diff_idx = np.argmin(np.absolute(likelihoods - likelihood)) + 1
            likelihood_new = likelihoods[likelihood_diff_idx - 1]
            data_index = size - likelihood_diff_idx - i - 1

            predicted_change = (self.data[data_index, :] - self.data[data_index - 1, :]) * np.sign(
                likelihood - likelihood_new)
            predicted = np.vstack((predicted, self.data[size - i - 1, :] + predicted_change))

            if i == self.window - 1:
                # first iteration disable hmm params initialization for next iterations
                hmm.init_params = ''

        return predicted

    def _find_optimal_states(self):
        pass

    def get_mean_error(self):
        pass

    def get_r2(self):
        pass
