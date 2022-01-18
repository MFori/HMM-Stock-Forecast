import logging

#from hmm_stock_forecast.base2 import Hovno
from hmm_stock_forecast.data.data import read_data
from hmm_stock_forecast.hmm.model import HMMStockForecastModel
from hmm_stock_forecast.plot.plot import show_plot
from hmm_stock_forecast.utils.args import parse_args
from hmm_stock_forecast.utils.logging import init_logger
import numpy as np

def main():
    init_logger()
    logging.info("App started")

 #   h = Hovno()
    args = parse_args()
    data = read_data(args)

    model = HMMStockForecastModel(data)
    predicted = model.run()[:, 2]

    show_plot(np.append((data[-200:, 2]), None), predicted, args.ticker if args.ticker else args.file, args.start)
    # show_plot(np.append(np.flipud(data[range(100), 2])[1:], None), predicted,
    #           args.ticker if args.ticker else args.file, args.start)
    # show_plot(np.append(np.flipud(data[range(100 + 1), 2])[:100], None), predicted,
    #           args.ticker if args.ticker else args.file, args.start)


if __name__ == '__main__':
    main()
