import sys
import numpy as np
from .data import get_data_csv
from .linearregression import LinearRegression


def ft_training(argv):

    file_name = get_args(argv)
    df = get_data_csv(file_name)
    lr = LinearRegression()
    lr.setData(df.values)
    lr.normData()
    lr.train()
    print(lr.getParams())
    np.savetxt("thetas.txt", lr.getParams())

    return 0


def get_args(argv):
    if len(argv) != 2:
        print("Usage: python3 ft_training.py myhappydata.csv", file=sys.stderr)
        sys.exit(1)

    return argv[1]


if __name__ == "__main__":
    raise SystemExit(ft_training(sys.argv))
