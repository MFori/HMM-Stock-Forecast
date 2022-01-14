import argparse

from recordclass import RecordClass


class Args(RecordClass):
    ticker: str
    file: str
    start: str
    end: str
    pass


def parse_args() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--ticker", type=str, help="Stock ticker, data will be downloaded from yahoofinance")
    parser.add_argument("-f", "--file", type=str, help="Data file")
    parser.add_argument("-s", "--start", type=str, help="Start date (used only with ticker), yyyy-mm-dd")
    parser.add_argument("-e", "--end", type=str, help="End date (used only with ticker), yyyy-mm-dd")

    a = parser.parse_args()

    args = Args(
        a.ticker,
        a.file,
        a.start,
        a.end
    )

    return args
