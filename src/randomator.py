import numpy as np
import json
import math
import sys

from .model import Model
from . import stats
from .config import *


class Randomator:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.nb_points = int(self.rng.integers(RAND_NB_POINTS_MIN, RAND_NB_POINTS_MAX))
        self.initModel()
        self.createData()
        self.addNoise()

    def save(self):
        self.setFileName()
        np.savetxt(
            self.file + ".csv",
            self.data.T.astype(int),
            fmt="%d",
            delimiter=",",
            header="mileage,price",
        )
        r_squared = math.nan
        try:
            r_squared = stats.r_squared(self.data.T, self.model)
        except ValueError as e:
            print("Value Error during R squared calculation\n", e, file=sys.stderr)

        meta = {
            "theta0": self.model.params[THETA_ZERO],
            "theta1": self.model.params[THETA_ONE],
            "nb_points": self.nb_points,
            "min_X": int(self.model.norm_mins[X]),
            "max_X": int(self.model.norm_mins[X] + self.model.norm_range[X]),
            "min_Y": int(self.model.norm_mins[Y]),
            "max_Y": int(self.model.norm_mins[Y] + self.model.norm_range[Y]),
            "noise_ratio": float(self.sigma[0]),
            "R2": r_squared,
        }
        print(meta)
        with open(self.file + ".json", "w") as f:
            json.dump(meta, f, indent=4)

    def setFileName(self):
        self.file = "randata" + str(self.rng.integers(0, RAND_MAX_FILE_NUMBER)).zfill(
            RAND_FILE_DIGITS
        )

    def initModel(self):
        self.model = Model(self.rng.uniform(0, 1, 2))
        log_max = self.rng.integers(RAND_DATA_DIGITS_MIN, RAND_DATA_DIGITS_MAX, 2)
        norm_max = self.rng.uniform(1, 10**log_max, 2)
        norm_min = norm_max * self.rng.uniform(0, RAND_DATA_MIN_MAX_RATIO_MAX, 2)
        self.model.norm_mins = np.array(norm_min)
        self.model.norm_range = np.array(norm_max - norm_min)
        self.model.normalized = True
        self.model.setParams(self.model.getRawParams())
        self.model.normalized = False

    def createData(self):
        self.data = np.zeros((2, self.nb_points))
        self.data[X] = self.rng.uniform(
            self.model.norm_mins[X],
            self.model.norm_mins[X] + self.model.norm_range[X] + 1,
            self.nb_points,
        )
        self.data[Y] = [self.model.eval(x) for x in self.data[X]]

    def addNoise(self):
        self.sigma = self.rng.uniform(RAND_NOISE_RATIO_MIN, RAND_NOISE_RATIO_MAX, 1)
        variance = stats.variance(self.data)
        self.data[Y] = self.rng.normal(self.data[Y], self.sigma * math.sqrt(variance))


if __name__ == "__main__":
    r = Randomator()
    r.save()
