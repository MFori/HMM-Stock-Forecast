import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CRITERIA = ['AIC', 'BIC', 'HQC', 'CAIC']


def show_plot(real, predicted, title, start):
    dates = pd.date_range(start, periods=len(real), freq="B")  # B freq for business days (skip weekends)
    df = pd.DataFrame({"Date": dates, "Real": real, "Predicted": predicted})

    plt.figure()
    ax = plt.gca()
    df.plot(kind="line", x="Date", y="Real", color="black", ax=ax, label="Actual Closing Price")
    df.plot(kind="line", x="Date", y="Predicted", color="red", ax=ax, label="Predicted Closing Price")
    plt.ylabel("Closing Price (USD)")
    plt.title("Closing Price of " + title)
    plt.legend(loc="upper left")
    plt.show()


def plot_criteria(stats):
    criteria = np.empty((4, 5, 50))

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
