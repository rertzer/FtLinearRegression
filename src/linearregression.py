import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .model import Model
from .config import *
from .trainingstats import TrainingStats
from . import stats


class LinearRegression:
    def __init__(self, learning_step=DEFAULT_STEP):
        self.model = Model()
        self.learning_step = learning_step
        self.data = None
        self.loops = LOOPS

    def setData(self, data):
        self.data = np.array(data)

    def normData(self):
        if self.data is None:
            raise ValueError("data can't be None")
        self.raw_data = self.data.copy()
        self.data = self.model.normData(self.data)

    def getParams(self):
        return self.model.getRawParams()

    def getPredictedDistance(self):
        if self.data is None:
            raise ValueError("data can't be None")
        return self.model.eval(self.data[:, X]) - self.data[:, Y]

    def getDelta(self):
        if self.data is None:
            raise ValueError("data can't be None")
        gap = self.getPredictedDistance()

        delta_zero = self.learning_step * gap.sum() / gap.size

        gap = gap * self.data[:, X]
        delta_one = self.learning_step * gap.sum() / gap.size

        return np.array((delta_zero, delta_one))

    def getLoss(self):
        if self.data is None:
            raise ValueError("data can't be None")
        squares = self.getPredictedDistance() ** 2
        return squares.sum() / len(self.data)

    def getRawLoss(self):
        raw_model = Model(self.getParams())
        squares = (raw_model.eval(self.raw_data[:, X]) - self.raw_data[:, Y]) ** 2

        return squares.sum() / len(self.raw_data)

    def getRsquared(self):
        return stats.r_squared(self.data, self.model)

    def getSSE(self):
        return stats.sse(self.data, self.model)

    def train(self):
        self.stats = TrainingStats(self.loops)
        for i in range(self.loops):
            self.model.updateParams(self.getDelta())
            self.stats.records[LOSS][i] = self.getLoss()
            self.stats.records[RSQUARED][i] = self.getRsquared()

    def print_results(self):
        p = self.getParams()
        print(f"Parameters: Theta 0: {p[0]}, Theta 1: {p[1]}")
        print(f"Normalized Loss: {self.stats.getLoss()}")
        print(f"Loss: {self.getRawLoss()}")
        print(f"R²: {self.stats.getRsquared()}")

    def plot(self):
        self.plot_norm_data()
        self.plot_raw_data()
        self.stats.plot()

    def plot_norm_data(self):
        df = pd.DataFrame(self.data, columns=["km", "price"])
        plt = self.plot_data(df)
        self.plot_theta(plt, df, self.model.params)

        plt.savefig(NORMALIZED_FILE)
        plt.close()

    def plot_raw_data(self):
        df = pd.DataFrame(self.raw_data, columns=["km", "price"])
        plt = self.plot_data(df)
        self.plot_theta(plt, df, self.getParams())

        plt.savefig(PLOT_FILE)
        plt.close()

    def plot_data(self, data):
        plt.scatter(data["km"], data["price"])
        plt.title("Car price per mileage")
        plt.xlabel("km")
        plt.ylabel("Price")

        return plt

    def plot_theta(self, plt, data, theta):
        line_x = np.linspace(data["km"].min(), data["km"].max(), 100)
        line_y = theta[THETA_ZERO] + theta[THETA_ONE] * line_x
        plt.plot(line_x, line_y, label="estimation")
        plt.legend()
