import numpy as np
from model import Model


data = ((1, 2), (3, 3), (5, 18))
data = np.array(data)

# print(data.shape)
# print(data[0])
# print(data[1])
# print(data[:, 0])

model = Model((1, 2))

print(data)
print(model.eval(data[:, 0]))
