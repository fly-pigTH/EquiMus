# load kick_action.npy and analysis
import numpy as np
import matplotlib.pyplot as plt
data = np.load('../data/kick_action.npy', allow_pickle=True)
print(data.shape)
plt.plot(data[:,0])
plt.plot(data[:,1])