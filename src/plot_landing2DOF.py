# 绘制静态力, 使用三维图像
import numpy as np
from matplotlib import pyplot as plt
import scipy
import math

# load the data
F1_matrix = scipy.io.loadmat('./src/F1_landing2DOF.mat')['F1_matrix']
F2_matrix = scipy.io.loadmat('./src/F2_landing2DOF.mat')['F2_matrix']

# 产生理论力的解
theta_1_array = np.linspace(math.pi/6, math.pi*2/3, 20) # 切分十个点
theta_2_array = np.linspace(0, math.pi*1/2, 20)

# 筛掉theta<math.pi/2的
mask = theta_1_array > math.pi/2
theta_1_array = theta_1_array[mask]

x, y = np.meshgrid(theta_2_array, theta_1_array)    # 这里调换顺序，保证theta_1_array 对应行

# 创建图像
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维表面
surf = ax.plot_surface(x*180/math.pi, y*180/math.pi, F1_matrix[mask, :], cmap='viridis', edgecolor='none', label="MAA")
surf = ax.plot_surface(x*180/math.pi, y*180/math.pi, F2_matrix[mask, :], cmap='coolwarm', edgecolor='none', label="BAA")

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=10)


# 设置标签
ax.set_title("3D Height Map", fontsize=14)
ax.set_xlabel("Theta_2/deg")
ax.set_ylabel("Theta_1/deg")
ax.set_zlabel("Force/N")
plt.legend()

# 显示图像
plt.show()