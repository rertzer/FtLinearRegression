import numpy as np
from .model import Model
from .utils import *


class LinearRegression:
    def __init__(self, learning_step=DEFAULT_STEP):
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
            theta_one = params[THETA_ONE] * self.norm_range[Y] / self.norm_range[X]
            theta_zero = (
                self.norm_mins[Y]
                + params[THETA_ZERO] * self.norm_range[Y]
                - theta_one * self.norm_mins[Y]
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

    def train(self):
        for _ in range(LOOPS):
            self.model.updateParams(self.getDelta())
