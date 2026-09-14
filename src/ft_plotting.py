import sys
import numpy as np
import matplotlib.pyplot as plt
from .model import Model
from .data import get_data_csv, get_data_txt


def ft_plot(argv):
    data, theta = get_vars(argv)
    plot_data(data)
    if theta is not None:
        plot_theta(theta)


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

    print(data)
    plt.scatter(data["km"], data["price"])

    plt.title("Best Prices")
    plt.xlabel("km")
    plt.ylabel("Price")
    plt.savefig("best_prices.png")
    plt.close()
    # plt.show()


def plot_theta(theta):
    pass


if __name__ == "__main__":
    raise SystemExit(ft_plot(sys.argv))
