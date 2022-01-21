import argparse

from recordclass import RecordClass


class Args(RecordClass):
    ticker: str
    file: str
    start: str
    end: str
    window: int
    model: str
    pass


def parse_args() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--ticker", type=str, help="Stock ticker, data will be downloaded from yahoofinance")
    parser.add_argument("-f", "--file", type=str, help="Data file")
    parser.add_argument("-s", "--start", type=str, help="Start date (used only with ticker), yyyy-mm-dd")
    parser.add_argument("-e", "--end", type=str, help="End date (used only with ticker), yyyy-mm-dd")
    parser.add_argument("-w", "--window", type=str, help="Training window", default=50)
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
