import logging
from hmm_stock_forecast.data.data import read_data
from hmm_stock_forecast.hmm.model import HMMStockForecastModel
from hmm_stock_forecast.plot.plot import show_plot
from hmm_stock_forecast.utils.args import parse_args
from hmm_stock_forecast.utils.logging import init_logger
import numpy as np


def main():
    init_logger()
    logging.info("App started")

    args = parse_args()
    data = read_data(args)

    model = HMMStockForecastModel(data)
    predicted = model.run()[:, 2]

    show_plot(np.append((data[-50:, 2]), None), predicted, args.ticker if args.ticker else args.file, args.start)


if __name__ == '__main__':
    main()
