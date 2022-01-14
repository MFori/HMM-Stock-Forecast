import logging

from hmm_stock_forecast.data.data import read_data
from hmm_stock_forecast.utils.args import parse_args
from hmm_stock_forecast.utils.logging import init_logger


def main():
    init_logger()
    logging.info('App started')

    args = parse_args()
    read_data(args)


if __name__ == '__main__':
    main()
