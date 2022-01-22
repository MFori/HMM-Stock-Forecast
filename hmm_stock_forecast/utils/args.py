import argparse

from recordclass import RecordClass


class Args(RecordClass):
    """
    Args holder helper class
    """
    ticker: str  # stock ticker (must be available at yahoo finance)
    file: str  # file name
    start: str  # start date (yyyy-mm-dd)
    end: str  # end date (yyyy-mm-dd)
    window: int  # window size
    model: str  # model type ('HMM' or 'pomegranate')
    pass


def parse_args() -> Args:
    """
    parse args
    :return: args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--ticker", type=str, help="Stock ticker, data will be downloaded from yahoofinance")
    parser.add_argument("-f", "--file", type=str, help="Data file")
    parser.add_argument("-s", "--start", type=str, help="Start date (used only with ticker), yyyy-mm-dd", required=True)
    parser.add_argument("-e", "--end", type=str, help="End date (used only with ticker), yyyy-mm-dd", required=True)
    parser.add_argument("-w", "--window", type=str, help="Training window", default=120)
    parser.add_argument("-m", "--model", type=str, help="'HMM' (our implementation, default) / 'pomegranate'",
                        default='HMM')

    a = parser.parse_args()

    args = Args(
        a.ticker,
        a.file,
        a.start,
        a.end,
        a.window,
        a.model
    )

    return args
