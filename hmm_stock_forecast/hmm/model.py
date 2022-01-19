import numpy as np

from hmm_stock_forecast.pyhhmm.hmm import HMM

# number of hmm states to test (and choose) using criteria
TEST_STATES = [2, 3, 4, 5, 6]


class HMMStockForecastModel:
    _hmm: HMM
    predicted: np.array

    def __init__(self, data, window=50):
        self.data = data
        self.window = window
        # TODO validate if window is not too small for small dataset
        pass

    def run(self):
        states = self._find_optimal_states()
        print('optimal states: ' + str(states))

        size = len(self.data)
        predicted = np.empty([0, 4])
        #hmm = HiddenMarkovModel.from_samples(NormalDistribution, n_components=states, X=self.data[:self.window, :])
        hmm = HMM(
            # number of hidden states
            n_states=4,
            # number of distinct emissions
            n_emissions=4,
            # can be 'diagonal', 'full', 'spherical', 'tied'
            covariance_type='diagonal',
        )
        hmm.init_params(self.data[:self.window, :])

        for i in reversed(range(self.window + 1)):
            print(i)

            train = self.data[size - self.window - i:size - i, :]
            #hmm.fit(train)
            print('train')
            hmm.init_params(train)
            hmm.train([train])

            #likelihood = hmm.log_probability(train)
            likelihood = hmm.log_likelihood(train)
            likelihoods = []
            j = i + 1
            step = 0
            # todo remove step for speedtup testing
            while size - self.window - j > 0 and step < 20:
                obs = self.data[size - self.window - j:size - j, :]
                #likelihoods = np.append(likelihoods, hmm.log_probability(obs))
                likelihoods = np.append(likelihoods, hmm.log_likelihood(obs))
                step+=1
                j += 1

            likelihood_diff_idx = np.nanargmin(np.absolute(likelihoods - likelihood)) + 1
            likelihood_new = likelihoods[likelihood_diff_idx - 1]
            data_index = size - likelihood_diff_idx - i - 1

            predicted_change = (self.data[data_index, :] - self.data[data_index - 1, :]) * np.sign(
                likelihood - likelihood_new)
            predicted = np.vstack((predicted, self.data[size - i - 1, :] + predicted_change))

        return predicted

    def _find_optimal_states(self):
        size = len(self.data)
        state_likelihood = []

        for states in TEST_STATES:
            #hmm = HiddenMarkovModel.from_samples(NormalDistribution, n_components=states, X=self.data[:self.window, :])
            hmm = HMM(
                # number of hidden states
                n_states=4,
                # number of distinct emissions
                n_emissions=4,
                # can be 'diagonal', 'full', 'spherical', 'tied'
                covariance_type='diagonal',
            )
            hmm.init_params(self.data[:self.window, :])
            offset = 0
            likelihoods = []
            invalid_states = False

            while offset + self.window <= size:
                data = self.data[offset:offset + self.window, :]
                hmm.train([data])
                try:
                    likelihoods = np.append(likelihoods, hmm.log_likelihood(data))
                except ValueError:
                    invalid_states = True
                    break
                offset += self.window

            if invalid_states:
                state_likelihood = np.append(state_likelihood, np.finfo(float).max)
                print('states: ' + str(states) + ' likelihood=invalid')
            else:
                likelihood = np.average(likelihoods)
                print('states: ' + str(states) + ' likelihood=' + str(likelihood))
                state_likelihood = np.append(state_likelihood, likelihood)

        return TEST_STATES[np.nanargmin(state_likelihood)]

    def get_mean_error(self):
        pass

    def get_r2(self):
        pass
