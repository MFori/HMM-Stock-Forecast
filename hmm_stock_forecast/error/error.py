from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score


def mean_absolute_percentage_error(actual, forecast) -> float:
    """
    calculate mean absolute percentage error (MAPE)
    :param actual: actual values
    :param forecast: forecast (prediction) values
    :return: MAPE
    """
    return mape(actual, forecast) * 100


def r_squared(actual, forecast) -> float:
    """
    calculate r2 (coefficient of determination)
    :param actual: actual values
    :param forecast: forecast (prediction) values
    :return: r2
    """
    return r2_score(actual, forecast)
