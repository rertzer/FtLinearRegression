import numpy as np
import pandas as pd
from .model import Model
from .config import *
from .ft_plotting import *
from .stats import Stats


class LinearRegression:
    def __init__(self, learning_step=DEFAULT_STEP):
        self.model = Model()
        self.learning_step = learning_step
        self.normalized = False
        self.data = None
        self.loops = LOOPS

    def setData(self, data):
        self.data = np.array(data)
        self.setSST()

    def normData(self):
        if self.data is not None:
            self.norm_mins = np.min(self.data, axis=0)
            self.norm_range = np.max(self.data, axis=0) - self.norm_mins
            self.data = (self.data - self.norm_mins) / self.norm_range
            self.setSST()
            self.normalized = True

    def getParams(self):
        params = self.model.params.copy()
        if self.normalized is True:
            theta_one = params[THETA_ONE] * self.norm_range[Y] / self.norm_range[X]
            theta_zero = (
                self.norm_mins[Y]
                + params[THETA_ZERO] * self.norm_range[Y]
                - theta_one * self.norm_mins[X]
            )

            params[THETA_ZERO] = theta_zero
            params[THETA_ONE] = theta_one
        return params

    def getPredictedDistance(self):
        assert self.data is not None
        return self.model.eval(self.data[:, X]) - self.data[:, Y]

    def getDelta(self):
        gap = self.getPredictedDistance()

        delta_zero = self.learning_step * gap.sum() / gap.size

        assert self.data is not None
        gap = gap * self.data[:, X]
        delta_one = self.learning_step * gap.sum() / gap.size

        return np.array((delta_zero, delta_one))

    def getLoss(self):
        assert self.data is not None
        squares = self.getPredictedDistance() ** 2
        return squares.sum() / len(self.data)

    def getRsquared(self):
        return self.getSSE() / self.sst

    def setSST(self):
        assert self.data is not None
        self.y_mean = self.data[:, Y].mean()
        y_dist_squared = (self.data[:, Y] - self.y_mean) ** 2
        self.sst = y_dist_squared.sum()

    def getSSE(self):
        assert self.data is not None
        squares = (self.model.eval(self.data[:, X]) - self.y_mean) ** 2
        return squares.sum()

    def train(self):
        self.stats = Stats(self.loops)
        for i in range(self.loops):
            self.model.updateParams(self.getDelta())
            self.stats.records[LOSS][i] = self.getLoss()
            self.stats.records[RSQUARED][i] = self.getRsquared()

    def plot(self):
        self.plot_norm_data()
        plot_stats(self.stats.records)

    def plot_norm_data(self):
        df = pd.DataFrame(self.data, columns=["km", "price"])
        plt = plot_data(df)
        plot_theta(plt, df, self.model.params)

        plt.savefig("normalized.png")
        plt.close()
