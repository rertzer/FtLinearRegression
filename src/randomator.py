import numpy as np

from .model import Model


class randomator:
    def __init__(self):
        rng = np.random.default_rng()
        self.file = "randata" + str(rng.integers(0, 1000000000)).zfill(9)
        self.nb_points = rng.integers(6, 667)
        self.model = Model(rng.uniform(0, 1, 2))
        norm_max = rng.uniform(0, 666666, 2)
        norm_min = norm_max * rng.uniform(0, 0.5, 2)
        self.model.norm_mins = norm_min
        self.model.norm_range = norm_max - norm_min
