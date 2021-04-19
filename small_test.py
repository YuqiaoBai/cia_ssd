import matplotlib.pyplot as plt
import numpy as np

map_bin = np.load('./data/map_colsed.npy').astype(int)
plt.imshow(map_bin)
plt.show()