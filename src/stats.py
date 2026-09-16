import numpy as np
from .config import *


class Stats:
    def __init__(self, loops):
        self.loops = loops
        self.records = np.zeros((2, loops))

    def getLoss(self):
        return self.records[LOSS][-1]

    def getRsquared(self):
        return self.records[RSQUARED][-1]
