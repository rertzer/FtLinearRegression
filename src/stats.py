import pandas as pd
import numpy as np


class Stats:
    def __init__(self, loops):
        self.loops = loops
        self.records = np.zeros((2, loops))
        # self.records = pd.DataFrame(0, index=range(loops), columns=["loss", "Rsquared"])
