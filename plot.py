# read sensor_list.npy and time_list.npy and plot
# 静力学结果绘制
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import scipy.io

folder_path = "Exp-SingleLeg-Super-20241213_072017"
RobotStateName = os.path.join(folder_path, "StaticState_list.npy")
StaticState = np.load(f"./data/{RobotStateName}")
StaticState.shape



# 创建图形和三维轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
# surf = ax.plot(StaticState[:, 0], StaticState[:, 1], StaticState[:, 2])
# surf = ax.plot(StaticState[:, 0], StaticState[:, 1], StaticState[:, 3])

# plt.show()


## 静力学绘制
# read the file
# MAAP = scipy.io.loadmat('F1.mat').get('F1')
# BAAP = scipy.io.loadmat('F2.mat').get('F2')

MAAP = scipy.io.loadmat('F1kkk.mat').get("F1_matrix") # new data
BAAP = scipy.io.loadmat('F2kkk.mat').get("F2_matrix")
print(MAAP.shape)

theta_1_array = np.linspace(0, math.pi*2/3, 100)
print(theta_1_array.shape)
print(f"delta = {theta_1_array[1] - theta_1_array[0]} {math.pi*2/3/99}")
theta_2_array = np.linspace(0, math.pi*2/3, 100)

X,Y = np.meshgrid(theta_1_array, theta_2_array) # X和Y需要是二维数组
print(X)
print(X[30:100][30:100].shape)

# ax.scatter(X[30:100, 30:100], Y[30:100, 30:100], MAAP[30:100, 30:100], s=1)
# ax.scatter(X[30:100, 30:100], Y[30:100, 30:100], BAAP[30:100, 30:100], s=1)

# plt.xlabel('theta_1')
# plt.ylabel('theta_2')
# plt.title('AAP')
# plt.show()

## 在相同的力作用下，对比q_sim和q_theo
# 直接绘制 q_theo VS （q_sim-q_theo）


# load sim data
# sim_data = np.load("./data/Exp-Static-20241220_152829/StaticState_list.npy")
sim_data = np.load("./data/Exp-Static_ideal-20241221_152046/StaticState_list.npy")

print(sim_data)
print("--")
size = 20
sim_data = sim_data.reshape(size,size,10, order="F")
print(sim_data[0,0,:])
theta_1_array = np.linspace(math.pi/6, math.pi*2/3, size)
theta_2_array = np.linspace(0, math.pi/2, size)
X,Y = np.meshgrid(theta_1_array, theta_2_array) # X和Y需要是二维数组
print(MAAP[0::11, 0::11].shape)
# ax.scatter(X, Y, MAAP[0::11, 0::11], s=10)
ax.scatter(X, Y, sim_data[:,:,2]+math.pi/2, s=4, label="the1_sim") # shoulder
ax.scatter(X, Y, X, s=4, label="the1_theo") # shoulder
ax.scatter(X, Y, sim_data[:,:,3], s=4, label="the2_sim") # elbow
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


import numpy as np
import matplotlib.pyplot as plt

# 计算误差矩阵（绝对误差）
error_matrix = np.abs(sim_data[:,:,2]+math.pi/2 - X)

# 绘制热力图
plt.figure(figsize=(12, 8))
plt.imshow(error_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Error')  # 添加颜色条
plt.clim(0, 0.2)
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
error_matrix = np.abs(sim_data[:,:,3] - Y)

# 绘制热力图
plt.figure(figsize=(12, 8))
plt.imshow(error_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Error')  # 添加颜色条
plt.clim(0, 0.2)
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