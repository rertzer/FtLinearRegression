import numpy as np

from .config import *


class Model:
    def __init__(self, params=(0, 0)):
        self.setParams(params)
        self.normalized = False

    def eval(self, value):
        return self.params[THETA_ZERO] + value * self.params[THETA_ONE]

    def setParams(self, params):
        params = np.array(params)
        if len(params) != 2:
            raise ValueError("Model accepts only 2 parameters")
        self.params = params

    def updateParams(self, update):
        self.params = self.params - np.array(update)

    def normData(self, data):
        if data is not None:
            self.norm_mins = np.min(data, axis=0)
            self.norm_range = np.max(data, axis=0) - self.norm_mins
            data = (data - self.norm_mins) / self.norm_range
            self.normalized = True
        return data

    def getRawParams(self):
        raw_params = self.params.copy()
        if self.normalized is True:
            raw_params[THETA_ONE] = self.getRawX(raw_params[THETA_ONE])
            raw_params[THETA_ZERO] = self.getRawY(
                raw_params[THETA_ONE], raw_params[THETA_ZERO]
            )

        return raw_params

    def getRawX(self, x):
        return x * self.norm_range[Y] / self.norm_range[X]

    def getRawY(self, raw_x, y):
        return self.norm_mins[Y] + y * self.norm_range[Y] - raw_x * self.norm_mins[X]
