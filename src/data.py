import sys
import pandas as pd
import numpy as np


def get_data_csv(file_name):
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


def get_data_txt(file_name):
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
