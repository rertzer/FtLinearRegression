from .config import *


def sst(data):
    if data is None:
        raise ValueError("data can't be None")
    y_dist_squared = (data[:, Y] - y_mean(data)) ** 2
    return y_dist_squared.sum()


def sse(data, model):
    if data is None or model is None:
        raise ValueError("sse arguments can't be None")
    squares = (model.eval(data[:, X]) - y_mean(data)) ** 2
    return squares.sum()


def y_mean(data):
    if data is None:
        raise ValueError("data can't be None")
    return data[:, Y].mean()


def r_squared(data, model):
    return sse(data, model) / sst(data)


def variance(data):
    if data is None or len(data) == 0:
        raise ValueError("Invalid data")
    return sst(data) / len(data)
