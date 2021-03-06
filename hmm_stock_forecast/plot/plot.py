import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CRITERIA = ['AIC', 'BIC', 'HQC', 'CAIC']


def show_plot(real, predicted, title, end) -> None:
    """
    Plot real and predicted values
    :param real: 1d array of real closing prices (last value None)
    :param predicted: 1d array of predicted closing prices (last value prediction for tomorrow)
    :param title: plot title
    :param end: end date (yyyy-mm-dd)
    """
    dates = pd.date_range(end=end, periods=len(real), freq="B")  # B freq for business days (skip weekends)
    df = pd.DataFrame({"Date": dates, "Real": real, "Predicted": predicted})

    plt.figure()
    ax = plt.gca()
    df.plot(kind="line", x="Date", y="Real", color="black", ax=ax, label="Actual Closing Price")
    df.plot(kind="line", x="Date", y="Predicted", color="red", ax=ax, label="Predicted Closing Price")
    plt.ylabel("Closing Price (USD)")
    plt.title("Closing Price of " + title)
    plt.legend(loc="upper left")
    plt.show(block=False)


def plot_criteria(stats) -> None:
    """
    Plot criteria stats
    :param stats: stats from forecast.find_optimal_states
    """
    sh = np.shape(stats)
    criteria = np.empty((sh[1], sh[0], sh[2]))

    for idx, state in enumerate(stats):
        criteria[0][idx] = state[0]
        criteria[1][idx] = state[1]
        criteria[2][idx] = state[2]
        criteria[3][idx] = state[3]

    for idx, c in enumerate(criteria):
        plt.figure()
        plt.title(CRITERIA[idx])
        for s, d in enumerate(c):
            plt.plot(d, label=(str(s + 2) + '-states'))
        plt.legend()

    plt.show(block=False)
    plt.pause(1)
