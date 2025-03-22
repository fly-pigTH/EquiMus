# read sensor_list.npy and time_list.npy and plot
# 静力学结果绘制
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


# 绘制曲面

## 静力学绘制

## 在相同的力作用下，对比q_sim和q_theo
# 直接绘制 q_theo VS （q_sim-q_theo）

# load sim data
# sim_data = np.load("./data/Exp-Static-20241220_152829/StaticState_list.npy")
sim_data = np.load("./data/Exp-Static_ideal_gridScan-20241230_104248/StaticState_list.npy")
# sim_data = np.load("./data/Exp-Static_ideal_gridScan-20241227_101514/StaticState_list.npy")
theta1_rank = 4
theta2_rank = 5

F1 = sim_data[:, 0]
F2 = sim_data[:, 1]
theta_1_data = sim_data[:, 4]
theta_2_data = sim_data[:, 5]
plt.plot(F1, label="F1")
plt.plot(F2, label="F2")

plt.plot(theta_1_data, label="theta_1_sim")
plt.plot(theta_2_data, label="theta_2_sim")
plt.legend()
plt.show()

# 创建图形和三维轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print(sim_data)
print("--")
size = 20
sim_data = sim_data.reshape(size,size,10, order="F")
print(sim_data[0,0,:])
theta_1_array = np.linspace(math.pi/6, math.pi*2/3, size)
theta_2_array = np.linspace(0, math.pi/2, size)
X,Y = np.meshgrid(theta_1_array, theta_2_array) # X和Y需要是二维数组

ax.scatter(X, Y, sim_data[:,:,theta1_rank]+math.pi/2, s=4, label="the1_sim") # shoulder
ax.scatter(X, Y, X, s=4, label="the1_theo") # shoulder
ax.scatter(X, Y, sim_data[:,:,theta2_rank], s=4, label="the2_sim") # elbow
ax.scatter(X, Y, Y, s=4, label="the2_theo") # elbow

# 计算最大差距
diff = np.max(np.abs(sim_data[2:,:-2,3] - X[2:,:-2]))
print(diff)


# control Z

plt.xlabel('theta_1')
plt.ylabel('theta_2')
plt.legend()
plt.show()

# surf = ax.plot(StaticState[:, 0], StaticState[:, 1], StaticState[:, 2])



# 计算误差矩阵（绝对误差）
error_matrix = np.abs(sim_data[:,:,theta2_rank]+math.pi/2 - X)

# 绘制热力图
plt.figure(figsize=(12, 8))
plt.imshow(error_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Error')  # 添加颜色条
# plt.clim(0, 0.04)
plt.title('Error Heatmap of Theta1')
plt.xlabel('theta_1/rad')
plt.ylabel('theta_2/rad')

# 设置 x 和 y 轴刻度
plt.xticks(ticks=np.arange(error_matrix.shape[1]), labels=[f'{round(i, 2)}' for i in theta_1_array], rotation=90)
plt.yticks(ticks=np.arange(error_matrix.shape[0]), labels=[f'{round(i, 2)}' for i in theta_2_array])

# 搜索最大坐标
max_index = np.argmax(error_matrix)  # 返回最大值在数组展平后的索引
max_coords = np.unravel_index(max_index, error_matrix.shape)  # 将展平索引转换为多维坐标

# 显示值
for i in range(error_matrix.shape[0]):
    for j in range(error_matrix.shape[1]):
        if i==max_coords[0] and j==max_coords[1]:
            plt.text(j, i, f"{error_matrix[i, j]:.4f}", ha='center', va='center', color='red')
        # plt.text(j, i, f"{error_matrix[i, j]:.2f}", ha='center', va='center', color='white')
plt.show()

# 计算误差矩阵（绝对误差）
error_matrix = np.abs(sim_data[:,:,theta2_rank] - Y + math.pi/2)

# 绘制热力图
plt.figure(figsize=(12, 8))
plt.imshow(error_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Error')  # 添加颜色条
# plt.clim(0, 0.04)
plt.title('Error Heatmap of Theta2')
plt.xlabel('theta_1/rad')
plt.ylabel('theta_2/rad')

# 设置 x 和 y 轴刻度
plt.xticks(ticks=np.arange(error_matrix.shape[1]), labels=[f'{round(i, 2)}' for i in theta_1_array], rotation=90)
plt.yticks(ticks=np.arange(error_matrix.shape[0]), labels=[f'{round(i, 2)}' for i in theta_2_array])

# 搜索最大坐标
max_index = np.argmax(error_matrix)  # 返回最大值在数组展平后的索引
max_coords = np.unravel_index(max_index, error_matrix.shape)  # 将展平索引转换为多维坐标

# 显示值
for i in range(error_matrix.shape[0]):
    for j in range(error_matrix.shape[1]):
        if i==max_coords[0] and j==max_coords[1]:
            plt.text(j, i, f"{error_matrix[i, j]:.4f}", ha='center', va='center', color='white')

plt.show()

