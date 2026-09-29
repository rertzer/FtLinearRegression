import numpy as np

from .model import Model
from .config import *


class randomator:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.file = "randata" + str(self.rng.integers(0, 1000000000)).zfill(9)
        self.nb_points = int(self.rng.integers(6, 667))
        self.init_model()
        self.init_data()

    def init_model(self):
        self.model = Model(self.rng.uniform(0, 1, 2))
        norm_max = self.rng.uniform(1, 666667, 2)
        norm_min = norm_max * self.rng.uniform(0, 0.5, 2)
        self.model.norm_mins = np.array(norm_min)
        self.model.norm_range = np.array(norm_max - norm_min)
        self.model.normalized = True
        self.model.setParams(self.model.getRawParams())

    def init_data(self):
        self.data = np.zeros((2, self.nb_points))
        self.data[X] = self.rng.uniform(
            self.model.norm_mins[X],
            self.model.norm_mins[X] + self.model.norm_range[X] + 1,
            self.nb_points,
        )
        self.data[Y] = [self.model.eval(x) for x in self.data[X]]
