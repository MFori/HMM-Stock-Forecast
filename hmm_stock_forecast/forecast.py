import numpy as np
from pomegranate import HiddenMarkovModel, NormalDistribution
from tqdm import tqdm
from hmm_stock_forecast.hmm.criteria import aic_criteria, bic_criteria, hqc_criteria, caic_criteria
from hmm_stock_forecast.hmm.hmm import HMM
from hmm_stock_forecast.hmm.ihmm import IHMM

# number of hmm states to test (and choose) using criteria
TEST_STATES = [2, 3, 4, 5, 6]
CLOSE_INDEX = 3


class StockForecast:
    """
    Stock forecasting using HMM,
    based on "Hidden Markov Model for Stock Trading" (Nguyet Nguyen, 2018)
    """
    window: int
    model: str

    def __init__(self, window=50, model='HMM'):
        """
        Constructor method
        :param window: training window used to set size of observation data for training and testing
            model, data of this size will be predicted
        :param model: value of 'HMM' for our HMM implementation based on Nguyen
            or 'pomegranate' for using pomegranate HMM library
        """
        self.window = window
        self._model_type = model
        pass

    def predict(self, states, data) -> np.array:
        """
        predict closing prices for data range of self.window
        :param states: number of HMM states
        :param data: stock data - 2d array of ['Open', 'Low', 'High', 'Close']
        :return: 1D array of predicted close values
        """
        size = len(data)
        predicted = np.empty([0, 1])
        hmm = self._create_and_init_model(states, data[:self.window, :])

        for i in tqdm(reversed(range(self.window + 1)), total=self.window + 1):
            train = data[size - self.window - i:size - i, :]
            hmm.fit(train)

            likelihood = hmm.log_probability(train)
            likelihoods = []
            j = i + 1
            step = 0
            # todo remove step for speedtup testing
            while size - self.window - j > 0 and step < 20:
                obs = data[size - self.window - j:size - j, :]
                likelihoods = np.append(likelihoods, hmm.log_probability(obs))
                step += 1
                j += 1

            # find sequence with similar likelihood
            likelihood_idx = np.nanargmin(np.absolute(likelihoods - likelihood)) + 1
            likelihood_new = likelihoods[likelihood_idx - 1]
            data_index = size - likelihood_idx - i - 1

            # predict closing price Ot = Ot + (O_new_t+1 - O_new_t) * sign(P(O|model) - P(O_new|model))
            predicted_t1 = \
                data[size - i - 1, CLOSE_INDEX] \
                + (data[data_index, CLOSE_INDEX] - data[data_index - 1, CLOSE_INDEX]) \
                * np.sign(likelihood - likelihood_new)
            predicted = np.vstack((predicted, predicted_t1))

        return predicted

    def find_optimal_states(self, data) -> [int, np.array]:
        """
        find optimal HMM states based on sample data
        :param data: sample stock data - 2d array of ['Open', 'Low', 'High', 'Close']
        :return: [optimal states, criteria statistics]
        """
        size = len(data)
        criteria = np.empty((len(TEST_STATES), 4, self.window))
        score = np.zeros(len(TEST_STATES))

        for idx, states in tqdm(enumerate(TEST_STATES), total=len(TEST_STATES)):
            invalid_states = False

            aic_list = []
            bic_list = []
            hqc_list = []
            caic_list = []

            offset = size - self.window * 2 + 1
            hmm = self._create_and_init_model(states, data[offset:offset + self.window, :])

            while offset + self.window <= size:
                hmm.fit(data[offset:offset + self.window, :])
                try:
                    likelihood = hmm.log_probability(data)
                    aic_list = np.append(aic_list, aic_criteria(likelihood, states))
                    bic_list = np.append(bic_list, bic_criteria(likelihood, states, size))
                    hqc_list = np.append(hqc_list, hqc_criteria(likelihood, states, size))
                    caic_list = np.append(caic_list, caic_criteria(likelihood, states, size))
                except ValueError:
                    invalid_states = True
                    break
                offset += 1

            pad_width = (0, self.window - len(aic_list))
            criteria[idx][0] = np.pad(aic_list, pad_width)
            criteria[idx][1] = np.pad(bic_list, pad_width)
            criteria[idx][2] = np.pad(hqc_list, pad_width)
            criteria[idx][3] = np.pad(caic_list, pad_width)

            if invalid_states:
                score[idx] = np.average(np.finfo(float).max)
            else:
                score[idx] = np.average(criteria[idx])

        return TEST_STATES[np.nanargmin(score)], criteria

    def _create_and_init_model(self, n_states, sample) -> IHMM:
        """
        Create hmm model based on model_type ("HMM" / "pomegranate")
        and initialize it with sample data
        :param n_states: number of HMM states
        :param sample: stock data - 2d array of ['Open', 'Low', 'High', 'Close']
        :return: HMM model (our or pomegranate)
        """
        if self._model_type == 'pomegranate':
            return HiddenMarkovModel.from_samples(NormalDistribution, n_components=n_states, X=sample)
        else:
            hmm = HMM(n_states=n_states)
            hmm.init_params(sample=sample)
            return hmm
