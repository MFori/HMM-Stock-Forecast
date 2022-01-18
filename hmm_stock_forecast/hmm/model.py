from hmmlearn.hmm import GaussianHMM
import numpy as np

# number of hmm states to test (and choose) using criteria
TEST_STATES = [2, 3, 4, 5, 6]


class HMMStockForecastModel:
    _hmm: GaussianHMM
    predicted: np.array

    def __init__(self, data, window=200):
        self.data = data
        self.window = window
        # TODO validate if window is not too small for small dataset
        pass

    def run(self):
        states = self._find_optimal_states()
        print('optimal states: ' + str(states))

        size = len(self.data)
        predicted = np.empty([0, 4])
        hmm = self._hmm = GaussianHMM(n_components=states, algorithm='viterbi', init_params="stmc")

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

            if i == self.window:
                # first iteration disable hmm params initialization for next iterations
                hmm.init_params = ''

        return predicted

    def _find_optimal_states(self):
        size = len(self.data)
        state_likelihood = []

        for states in TEST_STATES:
            hmm = GaussianHMM(n_components=states, algorithm='viterbi', init_params='tmc')
            hmm.fit(self.data)
            hmm.init_params = ''
            offset = 0
            likelihoods = []
            invalid_states = False

            while offset + self.window <= size:
                data = self.data[offset:offset + self.window, :]
                hmm.fit(data)
                try:
                    likelihoods = np.append(likelihoods, hmm.score(data))
                except ValueError:
                    invalid_states = True
                    break
                if offset == 0:
                    hmm.init_params = ''
                offset += self.window

            if invalid_states:
                state_likelihood = np.append(state_likelihood, np.finfo(float).max)
                print('states: ' + str(states) + ' likelihood=invalid')
            else:
                likelihood = np.average(likelihoods)
                print('states: ' + str(states) + ' likelihood=' + str(likelihood))
                state_likelihood = np.append(state_likelihood, likelihood)

        return TEST_STATES[np.argmin(state_likelihood)]

    def get_mean_error(self):
        pass

    def get_r2(self):
        pass
