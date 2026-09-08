import numpy
from src.model import Model


class LinearRegression:
    def __init__(self, learning_step=0.01):
        self.model = Model()
        self.learning_step = learning_step

    def setData(self, data):
        self.data = numpy.array(data)

    def getPredictedGap(self):
        return 0
