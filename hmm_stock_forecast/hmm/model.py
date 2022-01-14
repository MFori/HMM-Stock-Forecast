from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
import numpy as np

NUM_TEST = 200
NUM_ITERS = 10000
K = 50


class HMMStockForecastModel:
    _hmm: GaussianHMM

    def __init__(self, train_window=50, test_size=0.33):
        self.train_window = train_window
        self.test_size = test_size
        pass

    def train(self, data):
        self._split_dataset(data)
        self._hmm = GaussianHMM(n_components=4, covariance_type='full', algorithm='viterbi')
        self._hmm.fit(self._train[NUM_TEST:, :])

    def get_predicted_data(self):
        predicted_stock_data = np.empty([0, self._train.shape[1]])

        hmm = GaussianHMM(n_components=4, covariance_type='full', algorithm='viterbi', init_params='')
        hmm.transmat_ = self._hmm.transmat_
        hmm.startprob_ = self._hmm.startprob_
        hmm.means_ = self._hmm.means_
        hmm.covars_ = self._hmm.covars_
        self._hmm = hmm

        for idx in reversed(range(NUM_TEST)):
            train_dataset = self._train[idx + 1:, :]
            test_data = self._train[idx, :]
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
            predicted_stock_data = np.vstack((predicted_stock_data, self._train[idx + 1, :] + predicted_change))
            print(len(predicted_stock_data))

        print(len(predicted_stock_data))
        return predicted_stock_data
        pass

    def get_mean_error(self):
        pass

    def get_r2(self):
        pass

    def _split_dataset(self, data):
        train_data, test_data = train_test_split(data, test_size=self.test_size, shuffle=False)

        self._train = train_data
        self._test = test_data

    def _find_optimal_states(self):
        pass
