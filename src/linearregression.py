import numpy as np
from .model import Model


class LinearRegression:
    def __init__(self, learning_step=0.01):
        self.model = Model()
        self.learning_step = learning_step
        self.normalized = False
        self.data = None

    def setData(self, data):
        self.data = np.array(data)

    def normData(self):
        if self.data is not None:
            self.norm_mins = np.min(self.data, axis=0)
            self.norm_range = np.max(self.data, axis=0) - self.norm_mins
            self.data = (self.data - self.norm_mins) / self.norm_range
            self.normalized = True

    def getParams(self):
        params = self.model.params.copy()
        if self.normalized is True:
            theta_one = params[1] * self.norm_range[1] / self.norm_range[0]
            theta_zero = (
                self.norm_mins[1]
                + params[0] * self.norm_range[1]
                - theta_one * self.norm_mins[1]
            )

            params[0] = theta_zero
            params[1] = theta_one
        return params

    def getPredictedDistance(self):
        assert self.data is not None
        return self.model.eval(self.data[:, 0]) - self.data[:, 1]

    def getDelta(self):
        assert self.data is not None
        gap = self.getPredictedDistance()
        delta_zero = self.learning_step * gap.sum() / gap.size
        gap = gap * self.data[:, 0]
        delta_one = gap.sum() / gap.size
        delta_one = self.learning_step * delta_one
        return np.array((delta_zero, delta_one))

    def train(self):
        for _ in range(10000):
            self.model.updateParams(self.getDelta())
