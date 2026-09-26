import sys
from .model import Model
from .data import get_data_txt


def ft_predict(argv):

    mileage, file_name = get_arguments(argv)
    params = get_data_txt(file_name)
    model = Model(params)
    print(params)
    print(
        f"For a {mileage} km mileage, the estimated price is {round(model.eval(mileage))}"
    )

    return 0


def get_arguments(argv):
    try:
        if len(argv) == 2:
            return (float(argv[1]), "thetas.txt")
        elif len(argv) == 3:
            return (float(argv[1]), argv[2])
        else:
            print(
                "Usage: python3 -m src.ft_predict 42000 <paramfile.txt>",
                file=sys.stderr,
            )
            sys.exit(1)
    except ValueError:
        print("Error: non numerical value", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    raise SystemExit(ft_predict(sys.argv))
