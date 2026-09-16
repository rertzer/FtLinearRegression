import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import *
from .data import get_data_csv, get_data_txt


def ft_plot(argv):
    data, theta = get_vars(argv)
    plt = plot_data(data)
    if theta is not None:
        plot_theta(plt, data, theta)

    plt.savefig("best_prices.png")
    plt.close()


def get_args(argv):
    if len(argv) == 2:
        return (argv[1], None)
    elif len(argv) == 3:
        return (argv[1], argv[2])
    else:
        print("Usage: python3 -m src.ft_plotting data.csv <theta.txt>")
        sys.exit(1)


def get_vars(argv):

    data_file, theta_file = get_args(argv)
    data = get_data_csv(data_file)
    theta = None
    if theta_file is not None:
        theta = np.array(get_data_txt(theta_file))
    return data, theta


def plot_data(data):

    plt.scatter(data["km"], data["price"])

    plt.title("Car price per mileage")
    plt.xlabel("km")
    plt.ylabel("Price")

    return plt


def plot_theta(plt, data, theta):
    line_x = np.linspace(data["km"].min(), data["km"].max(), 100)
    line_y = theta[THETA_ZERO] + theta[THETA_ONE] * line_x
    plt.plot(line_x, line_y, label="estimation")
    plt.legend()


def plot_stats(records):
    df = pd.DataFrame(records.T, columns=["loss", "R_squared"])

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


if __name__ == "__main__":
    raise SystemExit(ft_plot(sys.argv))
