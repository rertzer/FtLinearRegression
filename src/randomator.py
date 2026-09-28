import numpy as np


class randomator:
    def __init__(self):
        rng = np.random.default_rng()
        self.file = "randata" + str(rng.integers(0, 1000000000)).zfill(9)
        self.nb_points = rng.integers(6, 667)

        self.thetas = rng.uniform(0, 1, 2)
