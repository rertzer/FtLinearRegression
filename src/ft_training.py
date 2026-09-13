import sys
import pandas as pd
import numpy as np
from .linearregression import LinearRegression


def ft_training(argv):

    file_name = get_arguments(argv)
    df = get_data(file_name)
    lr = LinearRegression()
    lr.setData(df.values)
    lr.normData()
    lr.train()
    print(lr.getParams())
    np.savetxt("thetas.txt", lr.getParams())

    return 0


def get_arguments(argv):
    if len(argv) != 2:
        print("Usage: python3 ft_training.py myhappydata.csv", file=sys.stderr)
        sys.exit(1)

    return argv[1]


def get_data(file_name):
    try:
        df = pd.read_csv(file_name)
    except (
        FileNotFoundError,
        PermissionError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
    ) as e:
        print(f"Error reading {file_name}: {e}", file=sys.stderr)
        sys.exit(1)
    return df


if __name__ == "__main__":
    raise SystemExit(ft_training(sys.argv))
