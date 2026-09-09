import sys
import pandas as pd
import numpy as np


def ft_training(argv):
    print(argv)
    if len(argv) != 2:
        print("Usage: python3 ft_training.py myhappydata.csv")
        return 1

    file_name = argv[1]
    try:
        df = pd.read_csv(file_name)
    except (
        FileNotFoundError,
        PermissionError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as e:
        print(f"Error reading {file_name}: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(ft_training(sys.argv))
