import sys
import json
import numpy as np
from .data import get_data_csv
from .linearregression import LinearRegression
from .config import *


def ft_training(argv):

    file_name = get_args(argv)
    df = get_data_csv(file_name)
    lr = LinearRegression()
    lr.setData(df.values)
    lr.normData()
    lr.train()
    save_thetas(lr.getParams())
    lr.print_results()
    lr.plot()

    return 0


def get_args(argv):
    if len(argv) != 2:
        print("Usage: python3 ft_training.py myhappydata.csv", file=sys.stderr)
        sys.exit(1)

    return argv[1]


def save_thetas(params):
    thetas = {
        "theta0": params[THETA_ZERO],
        "theta1": params[THETA_ONE],
    }
    with open(THETA_FILE, "w") as f:
        json.dump(thetas, f, indent=4)


if __name__ == "__main__":
    raise SystemExit(ft_training(sys.argv))
