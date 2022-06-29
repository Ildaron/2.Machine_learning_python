# https://translated.turbopages.org/proxy_u/en-ru.ru.c276140c-62bcb205-e902ecf9-74722d776562/https/stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy

import numpy as np
from scipy.spatial import procrustes

a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
mtx1, mtx2, disparity = procrustes(a, b)
round(disparity)
