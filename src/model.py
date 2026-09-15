import numpy as np

from .config import *


class Model:
    def __init__(self, params=(0, 0)):
        self.setParams(params)

    def eval(self, value):
        return self.params[THETA_ZERO] + value * self.params[THETA_ONE]

    def setParams(self, params):
        params = np.array(params)
        if len(params) != 2:
            raise ValueError("Model accepts only 2 parameters")
        self.params = params

    def updateParams(self, update):
        self.params = self.params - np.array(update)
