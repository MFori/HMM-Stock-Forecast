import logging
from hmm_stock_forecast.data.data import read_data
from hmm_stock_forecast.error.error import mean_absolute_percentage_error, r_squared
from hmm_stock_forecast.forecast import StockForecast, CLOSE_INDEX
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
    predicted = forecast.predict(states, data)[:, 0]
    actual = data[-args.window:, CLOSE_INDEX]  # get only closing prices

    logging.info("Plotting prediction")
    show_plot(np.append(actual, None), predicted, args.ticker if args.ticker else args.file, args.end)

    logging.info("Calculation MAPE and r2")
    predicted = predicted[:-1]  # remove last predicted item - not in actual
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r_squared(actual, predicted)

    print('MAPE: ', mape)
    print('r2: ', r2)

    input("Press Enter to exit...")


if __name__ == '__main__':
    main()
