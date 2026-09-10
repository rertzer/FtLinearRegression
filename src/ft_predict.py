import sys
import pandas as pd
import numpy as np
from .model import Model


def ft_predict(argv):

    mileage, file_name = get_arguments(argv)
    params = get_data(file_name)
    model = Model(params)
    print(params)
    print(
        f"For a {mileage} km mileage, the estimated price is {round(model.eval(mileage))}"
    )

    return 0


def get_arguments(argv):
    if len(argv) == 2:
        return (float(argv[1]), "thetas.txt")
    elif len(argv) == 3:
        return (float(argv[1]), argv[2])
    else:
        print("Usage: python3 -m src.ft_predict 42000 <paramfile.txt>")
        sys.exit(1)

    return argv[1]


def get_data(file_name):
    try:
        params = np.loadtxt(file_name)
    except (
        FileNotFoundError,
        PermissionError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
    ) as e:
        print(f"Error reading {file_name}: {e}", file=sys.stderr)
        sys.exit(1)
    return params


if __name__ == "__main__":
    raise SystemExit(ft_predict(sys.argv))
