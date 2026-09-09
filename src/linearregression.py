import numpy as np
from src.model import Model


class LinearRegression:
    def __init__(self, learning_step=0.01):
        self.model = Model()
        self.learning_step = learning_step

    def setData(self, data):
        self.data = np.array(data)

    def getPredictedGap(self):
        return self.model.eval(self.data[:, 0]) - self.data[:, 1]

    def getDelta(self):
        gap = self.getPredictedGap()
        delta_zero = self.learning_step * gap.sum() / gap.size
        gap = gap * self.data[:, 0]
        delta_one = self.learning_step * gap.sum() / gap.size
        return np.array((delta_zero, delta_one))

    def train(self):
        for _ in range(10000):
            self.model.updateParams(self.getDelta())
