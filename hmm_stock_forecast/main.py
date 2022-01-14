import logging

from hmm_stock_forecast.data.data import read_data
from hmm_stock_forecast.plot.plot import show_plot
from hmm_stock_forecast.utils.args import parse_args
from hmm_stock_forecast.utils.logging import init_logger


def main():
    init_logger()
    logging.info("App started")

    args = parse_args()
    data = read_data(args)

    predicted = [x + 10 for x in data[:, 2]]

    show_plot(data[:, 2], predicted, args.ticker, args.start)


if __name__ == '__main__':
    main()
