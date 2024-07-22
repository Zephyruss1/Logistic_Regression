import numpy as np

dizi1 = np.array([1, 2, 3, 4])
dizi2 = np.array([3, 4, 5, 6])

birlesim_dizisi = np.union1d(dizi1, dizi2)
print(birlesim_dizisi)