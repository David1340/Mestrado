from funcoes_quanticas import *
import numpy as np
n = 9
indices = np.arange(150)
M = len(indices)
x = AAQ(n,M,indices)
s = medir(x)
print(s)