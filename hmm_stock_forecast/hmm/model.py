from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
import numpy as np

NUM_TEST = 200
NUM_ITERS = 10000
K = 50


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
            print("index " + str(likelihood_diff_idx) + " - " + str(data_index) + " - " + str(predicted_change))
            predicted = np.vstack((predicted, self.data[size - i - 1, :] + predicted_change))

            if i == self.window - 1:
                # first iteration disable hmm params initialization for next iterations
                hmm.init_params = ''
       
        return predicted

    def train(self):
        self._hmm = GaussianHMM(n_components=4, covariance_type='full', algorithm='viterbi')
        self._hmm.fit(self.data[NUM_TEST:, :])
        # self._hmm.fit(np.ones([4,4], dtype=np.double))

    def _find_optimal_states(self):
        pass

    def get_predicted_data(self):
        predicted_stock_data = np.empty([0, self.data.shape[1]])

        # hmm = GaussianHMM(n_components=4, covariance_type='full', algorithm='viterbi', init_params='')
        # hmm.transmat_ = self._hmm.transmat_
        # hmm.startprob_ = self._hmm.startprob_
        # hmm.means_ = self._hmm.means_
        # hmm.covars_ = self._hmm.covars_
        # self._hmm = hmm

        for idx in reversed(range(NUM_TEST)):
            train_dataset = self.data[idx + 1:, :]
            test_data = self.data[idx, :]
            num_examples = train_dataset.shape[0]
            # model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', startprob_prior=dirichlet_params, transmat_prior=dirichlet_params, tol=0.0001, n_iter=NUM_ITERS, init_params='mc')
            # if idx == NUM_TEST - 1:
            #    self._hmm = GaussianHMM(n_components=4, covariance_type='full')
            # elif idx == NUM_TEST - 2:
            #    # Retune the model by using the HMM paramters from the previous iterations as the prior
            #    self._hmm = GaussianHMM(n_components=4, covariance_type='full', init_params='')
            #    self._hmm.transmat_ = transmat_retune_prior
            #    self._hmm.startprob_ = startprob_retune_prior
            #    self._hmm.means_ = means_retune_prior
            #    self._hmm.covars_ = covars_retune_prior

            if idx == NUM_TEST - 1:
                self._hmm.init_params = ''

            self._hmm.fit(np.flipud(train_dataset))

            # transmat_retune_prior = self._hmm.transmat_
            # startprob_retune_prior = self._hmm.startprob_
            # means_retune_prior = self._hmm.means_
            # covars_retune_prior = self._hmm.covars_

            iters = 1
            past_likelihood = []
            curr_likelihood = self._hmm.score(np.flipud(train_dataset[0:K - 1, :]))

            while iters < num_examples / K - 1:
                past_likelihood = np.append(past_likelihood,
                                            self._hmm.score(np.flipud(train_dataset[iters:iters + K - 1, :])))
                iters = iters + 1
            likelihood_diff_idx = np.argmin(np.absolute(past_likelihood - curr_likelihood))
            predicted_change = train_dataset[likelihood_diff_idx, :] - train_dataset[likelihood_diff_idx + 1, :]
            predicted_stock_data = np.vstack((predicted_stock_data, self.data[idx + 1, :] + predicted_change))
            print(len(predicted_stock_data))

        print(len(predicted_stock_data))
        return predicted_stock_data
        pass

    def get_mean_error(self):
        pass

    def get_r2(self):
        pass
