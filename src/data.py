import sys
import json
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
    check_valid_data(df, file_name)

    return df


def get_data_txt(file_name):
    try:
        with open(file_name, "r") as f:
            data = json.load(f)
            params = np.array(data["theta0"], ["theta1"])
    except (
        FileNotFoundError,
        PermissionError,
        UnicodeDecodeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as e:
        print(f"Error reading {file_name}: {e}", file=sys.stderr)
        sys.exit(1)
    return params


def check_valid_data(df, file_name):
    if df.isna().any().any():
        print(f"Error parsing {file_name}: missing values", file=sys.stderr)
        sys.exit(1)
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
        print(f"Error reading {file_name}: non numeric values", file=sys.stderr)
        sys.exit(1)
