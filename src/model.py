class Model:
    def __init__(self, params=(0, 0)):
        params = list(params)
        if len(params) != 2:
            raise ValueError("Model accepts only 2 parameters")
        self.params = params

    def eval(self, value):
        return self.params[0] + value * self.params[1]
