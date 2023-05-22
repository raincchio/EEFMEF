import numpy as np
import math

beta=1

x = np.random.randint(20, size=(1, 10)) * -1

print(x)
data = np.exp(x*beta - x.max()*beta) / np.exp(x*beta - x.max()*beta).sum()
res = data*x
print(res.sum(), x.max())
