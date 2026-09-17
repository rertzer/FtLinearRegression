import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import *


class Stats:
    def __init__(self, loops):
        self.loops = loops
        self.records = np.zeros((2, loops))

    def getLoss(self):
        return self.records[LOSS][-1]

    def getRsquared(self):
        return self.records[RSQUARED][-1]

    def plot(self):
        df = pd.DataFrame(self.records.T, columns=["loss", "R_squared"])

        iterations = np.arange(len(df))

        fig, ax1 = plt.subplots()

        # Loss
        line1 = ax1.plot(iterations, df["loss"], color="blue", label="Loss")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")

        # R²
        ax2 = ax1.twinx()
        line2 = ax2.plot(iterations, df["R_squared"], color="red", label="R²")
        ax2.set_ylabel("R²")

        lines = line1 + line2

        ax1.legend(lines, ["Loss", "R²"])

        fig.suptitle("Training Statistics")
        plt.savefig("stats.png")
        plt.close()
