import numpy as np

a = np.array([1., -1., 0., 1.])

b = np.where(a < 0, 0., a)

print(b)