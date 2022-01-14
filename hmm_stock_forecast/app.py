import logging

from hmm_stock_forecast.data.data import read_data_from_yahoo, read_data_from_file, read_data
from hmm_stock_forecast.utils.args import parse_args
from hmm_stock_forecast.utils.logging import init_logger


class Application:
    def __init__(self):
        init_logger()
        super(Application, self).__init__()
        logging.info('App started')

        args = parse_args()
        read_data(args)
        print(args.ticker)
        read_data_from_yahoo(args.ticker, args.start, args.end)
        read_data_from_file(args.file)
