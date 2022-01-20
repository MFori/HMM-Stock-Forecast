import logging

import numpy as np
import pandas_datareader as pdr

from hmm_stock_forecast.utils.args import Args


def read_data(args: Args):
    data = None

    if args.ticker and args.start and args.end:
        logging.info(
            "Reading historical data of " + args.ticker + " from yahoo between " + args.start + " and " + args.end)
        data = read_data_from_yahoo(args.ticker, args.start, args.end)
        if data is None:
            logging.error('Could not read data from yahoo. Check if the ticker is valid.')
        else:
            logging.info('Data of ' + args.ticker + ' read successfully.')
    elif args.file:
        logging.info("Reading data from file " + args.file)
        data = read_data_from_file(args.file)
        if data is None:
            logging.error('Cant read data from file ' + args.file)
        else:
            logging.info('Data from file ' + args.file + ' raed successfully.')

    return data


def read_data_from_file(file_name):
    try:
        data = np.genfromtxt(file_name, delimiter=',')
        if data.shape[1] != 4:
            return None
        return data
    except Exception:
        return None


def read_data_from_yahoo(ticker, start, end):
    try:
        data = pdr.get_data_yahoo(ticker, start, end)

        open_price = np.array(data['Open'])
        low_price = np.array(data['Low'])
        high_price = np.array(data['High'])
        close_price = np.array(data['Close'])

        return np.column_stack((open_price, low_price, high_price, close_price))
    except Exception:
        return None
