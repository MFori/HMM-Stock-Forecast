import logging
from hmm_stock_forecast.data.data import read_data
from hmm_stock_forecast.forecast import StockForecast
from hmm_stock_forecast.plot.plot import show_plot, plot_criteria
from hmm_stock_forecast.utils.args import parse_args
from hmm_stock_forecast.utils.logging import init_logger
import numpy as np


def main():
    init_logger()
    logging.info("App started")

    args = parse_args()
    data = read_data(args)
    if data is None:
        return

    logging.info("Creating forecast model, window=" + str(args.window) + ", model=" + str(args.model))
    forecast = StockForecast(window=args.window, model=args.model)

    logging.info("Finding optimal states")
    states, stats = forecast.find_optimal_states(data)
    logging.info("Optimal states: " + str(states))

    logging.info("Plotting criteria stats")
    plot_criteria(stats)

    logging.info("Running prediction")
    predicted = forecast.run(states, data)[:, 2]

    logging.info("Plotting prediction")
    show_plot(np.append((data[-50:, 2]), None), predicted, args.ticker if args.ticker else args.file, args.end)

    # TODO calc error


if __name__ == '__main__':
    main()
