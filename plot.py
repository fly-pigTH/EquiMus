# read sensor_list.npy and time_list.npy and plot

import numpy as np
import matplotlib.pyplot as plt
import os
import math

folder_path = "Exp-SingleLeg-Super-20241213_072017"
RobotStateName = os.path.join(folder_path, "StaticState_list.npy")
StaticState = np.load(f"./data/{RobotStateName}")
StaticState.shape

# 创建图形和三维轴
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot(StaticState[:, 0], StaticState[:, 1], StaticState[:, 2])
surf = ax.plot(StaticState[:, 0], StaticState[:, 1], StaticState[:, 3])

plt.show()