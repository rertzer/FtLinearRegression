import sys
from .model import Model
from .getdata import get_data_json


def ft_predict(argv):

    file_name = get_arguments(argv)
    mileage = get_mileage()
    params = get_data_json(file_name)
    model = Model(params)
    print(
        f"For a {mileage} km mileage, the estimated price is {round(model.eval(mileage))}"
    )

    return 0


def get_arguments(argv):
    if len(argv) == 1:
        return "thetas.json"
    elif len(argv) == 2:
        return argv[1]
    else:
        print(
            "Usage: python3 -m src.ft_predict <paramfile.json>",
            file=sys.stderr,
        )
        sys.exit(1)


def get_mileage():
    try:
        mileage = float(input("Enter a car mileage: "))
    except ValueError as e:
        print(e)
        sys.exit(1)
    return mileage


if __name__ == "__main__":
    raise SystemExit(ft_predict(sys.argv))
