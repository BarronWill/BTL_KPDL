import numpy as np
from Data import *
import pandas as pd
a = np.random.randint(10, size=(2,3))
b = np.random.randint(10, size=(2,3))
print(a)
print('--------')
print(b)
# distance = np.min((a[:, np.newaxis] - b).sum(axis=2), axis=1)
print((a[:, np.newaxis] - b).sum(axis=2).mean(axis=0))
print((a[:, np.newaxis] - b).sum(axis=2))
print((a[:, np.newaxis] - b))

