import matplotlib.pyplot as plt
import pandas as pd


def show_plot(real, predicted, title, start):
    dates = pd.date_range(start, periods=len(real), freq="1D")
    df = pd.DataFrame({"Date": dates, "Real": real, "Predicted": predicted})

    ax = plt.gca()
    df.plot(kind="line", x="Date", y="Real", color="cyan", ax=ax, label="Actual Closing Price")
    df.plot(kind="line", x="Date", y="Predicted", color="red", ax=ax, label="Predicted Closing Price")
    plt.ylabel("Closing Price (USD)")
    plt.title("Closing Price of " + title)
    plt.legend(loc="upper left")
    plt.show()
