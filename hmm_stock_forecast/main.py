import logging

#from hmm_stock_forecast.base2 import Hovno

from hmm_stock_forecast.data.data import read_data
from hmm_stock_forecast.hmm.hmm import HMM
from hmm_stock_forecast.hmm.model import HMMStockForecastModel
from hmm_stock_forecast.plot.plot import show_plot
from hmm_stock_forecast.pyhhmm.gaussian import GaussianHMM
from hmm_stock_forecast.utils.args import parse_args
from hmm_stock_forecast.utils.logging import init_logger
import numpy as np
import pandas as pd

def test_2(V):
    my_hmm = GaussianHMM(
        n_states=4,
        n_emissions=2,
        covariance_type='diagonal'
    )

    # we set model parameters according to the example
    # the initial state probabilities, array of shape (n_states, )
    my_hmm.pi = np.array([0.6, 0.3, 0.1, 0.0])
    # the state transition probabilities, array of shape (n_states, n_states)
    my_hmm.A = np.array(
        [
            [0.7, 0.2, 0.0, 0.1],
            [0.3, 0.5, 0.2, 0.0],
            [0.0, 0.3, 0.5, 0.2],
            [0.2, 0.0, 0.2, 0.6],
        ]
    )
    # the means of each component
    my_hmm.means = np.array([[0.0, 0.0], [0.0, 11.0], [9.0, 10.0], [11.0, -1.0]])

    # the covariance of each component - shape depends `covariance_type`
    #             (n_states, )                          if 'spherical',
    #             (n_states, n_emissions)               if 'diagonal',
    #             (n_states, n_emissions, n_emissions)  if 'full'
    #             (n_emissions, n_emissions)            if 'tied'

    # generate observation sequences of different lengths
    lengths = np.random.randint(25, 150, size=25)
    #X = [
    #    my_hmm.sample(n_sequences=1, n_samples=n_samples)[0] for n_samples in lengths
    #]
    X = [V]
    print(X)

    # instantiate a MultinomialHMM object
    trained_hmm = GaussianHMM(
        # number of hidden states
        n_states=4,
        # number of distinct emissions
        n_emissions=4,
        # can be 'diagonal', 'full', 'spherical', 'tied'
        covariance_type='diagonal',
    )

    # reinitialise the parameters and see if we can re-learn them
    trained_hmm, log_likelihoods = trained_hmm.train(
        X,
        n_iter=100,   # maximum number of iterations to run
        conv_thresh=0.001,  # what percentage of change in the log-likelihood between iterations is considered convergence
        # whether to plot the evolution of the log-likelihood over the iterations
        # set to True if want to train until maximum number of iterations is reached
    )

def test(V):
    #data = pd.read_csv('data_python.csv.txt')

    #V = data['Visible'].values
    V = np.array([0.3, 1.0, 1.3, 2.4, 0.6, 1.1, 2.0, 0.0, 1.3, 0.1, 2.5])

    #V = [item for item in V]
    #V = np.array(V)
    #V = np.concatenate(V) #.reshape(-1, 1)

    # Transition Probabilities
    a = np.ones((2, 2))
    a = a / np.sum(a, axis=1)

    # Emission Probabilities
    b = np.array(((1, 3, 5, 4), (2, 4, 6, 3)))
    b = b / np.sum(b, axis=1).reshape((-1, 1))

    print(V)
    print(b)

    # Equal Probabilities for the initial distribution
    initial_distribution = np.array((0.5, 0.5))

    n_iter = 100
    hmm = HMM(2)
    hmm.init_params([])
    #a_model, b_model = hmm.baum_welch(V.copy(), n_iter=n_iter)
    #a_model, b_model = hmm.fit(V) #hmm.baum_welch(V, n_iter=n_iter)
    a_model, b_model = hmm.baum_welch(V, a, b, initial_distribution, n_iter=n_iter)
    print(f'Custom model A is \n{a_model} \n \nCustom model B is \n{b_model}')

def main():
    init_logger()
    logging.info("App started")

    args = parse_args()
    data = read_data(args)

    #test_2(data)
    #test(data)
    #return 0

 #   h = Hovno()

    model = HMMStockForecastModel(data)
    predicted = model.run()[:, 2]

    show_plot(np.append((data[-50:, 2]), None), predicted, args.ticker if args.ticker else args.file, args.start)
    # show_plot(np.append(np.flipud(data[range(100), 2])[1:], None), predicted,
    #           args.ticker if args.ticker else args.file, args.start)
    # show_plot(np.append(np.flipud(data[range(100 + 1), 2])[:100], None), predicted,
    #           args.ticker if args.ticker else args.file, args.start)


if __name__ == '__main__':
    main()
