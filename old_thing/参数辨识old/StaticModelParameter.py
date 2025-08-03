# 通过静力学试验数据和静力学理论模型，对参数进行辨识
# 悬挂状态
# 统一采用国际单位制

import numpy as np
import math
from scipy.optimize import minimize
import pandas as pd
import random

# a_1 = 0.25
# a_2 = 0.25
# b_1 = 0.21213
# b_2 = 0.1
# d_1 = 0.06
# d_2 = 0.10
# # s = 0

# # M = 0.155  # 真实值
# m_1, m_2 = 0.086, 0.1033
# # I_1, I_2 = (1/12 * m_1 * 0.25**2), (1/12 * m_2 * 0.25**2)

# m_3, m_4 = 0.18, 0.18

# # 刚度参数
# k_3, k_4 = 637.52/2, 631.6/2

# # 阻尼参数
# # c_3, c_4 = 22.68/2, 21.8/2

# # 重力加速度
# g = 9.8

# # 角度参数 (度转弧度)
# beta_1 = 8.13 / 180 * np.pi
# beta_2 = 30 / 180 * np.pi

# # 长度参数
# l_10 = 0.174
# l_20 = 0.252

# 作用面积
# S_1 = 6.54*1e-4
# S_2 = 6.37*1e-4

def get_static_func(theta_1, theta_2, params):

    # 参数
    a_1, a_2, b_1, b_2, d_1, d_2 = 0.25, 0.25, 0.21213, 0.1, 0.06, 0.10
    beta_1, beta_2 = 8.13 / 180 * np.pi, 30 / 180 * np.pi
    g = 9.8
    k_3, k_4, m_1, m_2, m_3, m_4, l_10, l_20, S_1, S_2 = params

    A_x_O = d_1
    A_y_O = 0

    B_x_O = -d_2
    B_y_O = 0

    C_x_O = b_1 * np.cos(theta_1 - beta_1)
    C_y_O = b_1 * np.sin(theta_1 - beta_1)

    D_x_O = a_1 * np.cos(theta_1) + b_2 * np.cos(theta_1 + theta_2 + beta_2)
    D_y_O = a_1 * np.sin(theta_1) + b_2 * np.sin(theta_1 + theta_2 + beta_2)

    E_x_O = a_1 * np.cos(theta_1)
    E_y_O = a_1 * np.sin(theta_1)

    F_x_O = a_1 * np.cos(theta_1) + a_2 * np.cos(theta_1 + theta_2)
    F_y_O = a_1 * np.sin(theta_1) + a_2 * np.sin(theta_1 + theta_2)

    # 计算偏导数
    # 对 theta_1 的偏导数

    dA_x_O_dtheta_1 = 0
    dA_y_O_dtheta_1 = 0

    dB_x_O_dtheta_1 = 0
    dB_y_O_dtheta_1 = 0

    dC_x_O_dtheta_1 = -b_1 * np.sin(theta_1 - beta_1)
    dC_y_O_dtheta_1 = b_1 * np.cos(theta_1 - beta_1)

    dD_x_O_dtheta_1 = -a_1 * np.sin(theta_1) - b_2 * np.sin(theta_1 + theta_2 + beta_2)
    dD_y_O_dtheta_1 = a_1 * np.cos(theta_1) + b_2 * np.cos(theta_1 + theta_2 + beta_2)

    dE_x_O_dtheta_1 = -a_1 * np.sin(theta_1)
    dE_y_O_dtheta_1 = a_1 * np.cos(theta_1)

    dF_x_O_dtheta_1 = -a_1 * np.sin(theta_1) - a_2 * np.sin(theta_1 + theta_2)
    dF_y_O_dtheta_1 = a_1 * np.cos(theta_1) + a_2 * np.cos(theta_1 + theta_2)

    # 对 theta_2 的偏导数

    dA_x_O_dtheta_2 = 0
    dA_y_O_dtheta_2 = 0

    dB_x_O_dtheta_2 = 0
    dB_y_O_dtheta_2 = 0

    dC_x_O_dtheta_2 = 0
    dC_y_O_dtheta_2 = 0

    dD_x_O_dtheta_2 = -b_2 * np.sin(theta_1 + theta_2 + beta_2)
    dD_y_O_dtheta_2 = b_2 * np.cos(theta_1 + theta_2 + beta_2)

    dE_x_O_dtheta_2 = 0
    dE_y_O_dtheta_2 = 0

    dF_x_O_dtheta_2 = -a_2 * np.sin(theta_1 + theta_2)
    dF_y_O_dtheta_2 = a_2 * np.cos(theta_1 + theta_2)

    # 计算长度 l1 和 l2
    l_1 = np.sqrt((A_x_O - C_x_O)**2 + (A_y_O - C_y_O)**2)
    l_2 = np.sqrt((B_x_O - D_x_O)**2 + (B_y_O - D_y_O)**2)

    # 计算偏导数
    # 偏导数 d/dtheta_1
    dl_1_dtheta_1 = 1/l_1 * ((A_x_O - C_x_O) * (dA_x_O_dtheta_1 - dC_x_O_dtheta_1) + (A_y_O - C_y_O) * (dA_y_O_dtheta_1 - dC_y_O_dtheta_1))
    dl_2_dtheta_1 = 1/l_2 * ((B_x_O - D_x_O) * (dB_x_O_dtheta_1 - dD_x_O_dtheta_1) + (B_y_O - D_y_O) * (dB_y_O_dtheta_1 - dD_y_O_dtheta_1))

    # 偏导数 d/dtheta_2
    dl_1_dtheta_2 = 1/l_1 * ((A_x_O - C_x_O) * (dA_x_O_dtheta_2 - dC_x_O_dtheta_2) + (A_y_O - C_y_O) * (dA_y_O_dtheta_2 - dC_y_O_dtheta_2))
    dl_2_dtheta_2 = 1/l_2 * ((B_x_O - D_x_O) * (dB_x_O_dtheta_2 - dD_x_O_dtheta_2) + (B_y_O - D_y_O) * (dB_y_O_dtheta_2 - dD_y_O_dtheta_2))

    # print(dl_1_dtheta_2)  # check the model

    # 等式右侧
    RHSb_1 = -( m_1*g*(dE_y_O_dtheta_1/2) + m_2*g*((dE_y_O_dtheta_1+dF_y_O_dtheta_1)/2) + m_3*g*(dC_y_O_dtheta_1/2) + m_4*g*(dD_y_O_dtheta_1/2) )
    RHSb_2 = -( m_1*g*(dE_y_O_dtheta_2/2) + m_2*g*((dE_y_O_dtheta_2+dF_y_O_dtheta_2)/2) + m_3*g*(dC_y_O_dtheta_2/2) + m_4*g*(dD_y_O_dtheta_2/2) )
    b = np.array([RHSb_1, RHSb_2])

    # 等式左侧
    LHSA = np.array([[dl_1_dtheta_1, dl_2_dtheta_1], 
                    [dl_1_dtheta_2, dl_2_dtheta_2]])

    # 回复力
    F_k = np.array([k_3*(l_1-l_10), k_4*(l_2-l_20)])

    # 计算静力学
    StaticForce = np.linalg.solve(LHSA, b) + F_k
    StaticP = np.array([StaticForce[0]/S_1, StaticForce[1]/S_2])
    return StaticP

def objective_function(params, theta_data, P_data):
    # 计算预测值
    P_predict = []
    # print("len(theta_data)", len(theta_data))
    for i in range(len(theta_data)):
        theta_1 = theta_data[i][0]
        theta_2 = theta_data[i][1]
        P_predict.append(get_static_func(theta_1, theta_2, params))
    P_predict = np.array(P_predict)
    # print(f"Shape of P_predict: {P_predict.shape}")
    # print(f"Shape of P_data: {P_data.shape}")
    # 计算损失
    loss = np.sum((P_data - P_predict)**2)
    return loss

# 初始参数猜测
initial_params = [637.52*2.8/2, 631.6/2, 0.086, 0.1033, 0.18, 0.18, 0.174, 0.252, 6.54*1e-4, 6.37*1e-4]
# initial_params = [0.25, 0.25, 0.21213, 0.1, 0.06, 0.10, 700/2, 700/2, 0.086, 0.1033, 0.18, 0.18, 9.8, 8.13 / 180 * np.pi, 30 / 180 * np.pi, 0.174, 0.252, 6.54*1e-4, 6.37*1e-4]



data = pd.read_csv("log/realdata/StaticProcess/11group.csv")
print(data)
# 实验数据
P1_array = data['P1'].values
P2_array = data['P2'].values
theta1_array = data['theta1'].values
theta2_array = data['theta2'].values
P_data = []
theta_data = []
for i in range(len(P1_array)):
    noise1 = random.uniform(-5000, 5000)*0
    noise2 = random.uniform(-5000, 5000)*0
    P_data.append([P1_array[i]*1000+noise1, P2_array[i]*1000+noise2])
    theta_data.append([theta1_array[i]*math.pi/180, theta2_array[i]*math.pi/180])
P_data = np.array(P_data)
theta_data = np.array(theta_data)


init_angle = [74.12572980809307, 60.829106796344746]
init_p = [0, 0]
print(get_static_func(init_angle[0], init_angle[1], initial_params))




# 原始数据计算：theta -> P
for i in range(len(P_data)):
    P_model = get_static_func(theta_data[i][0], theta_data[i][1], initial_params)
    print(f"Exper Res: Theta 1: {theta_data[i][0]}, Theta 2: {theta_data[i][1]}, P1: {P_data[i][0]}, P2: {P_data[i][1]}")
    print(f"Model Res: Theta 1: {theta_data[i][0]}, Theta 2: {theta_data[i][1]}, P1: {P_model[0]}, P2: {P_model[1]}")



print(f"len(theta_data): {len(theta_data)}")

# 设置优化选项
options = {
    'disp': True,          # 显示优化过程信息
    'maxiter': 100000,      # 最大迭代次数
    # 'ftol': 1e-8,          # 目标函数变化的容忍度
    # 'xtol': 1e-8           # 参数变化的容忍度
}

# 优化
result = minimize(objective_function, initial_params, args=(theta_data, P_data), method='L-BFGS-B', options=options)

# 初始loss
print("初始损失:", objective_function(initial_params, theta_data, P_data))
# 初始RMSE
print("初始RMSE:", np.sqrt(objective_function(initial_params, theta_data, P_data)/len(P_data)/2))

# 输出回归结果
optimal_params = result.x
# loss
print("回归得到的损失:", result.fun)
# MSE
print("回归得到的MSE:", result.fun/len(P_data)/2)
# RMSE
print("回归得到的RMSE:", np.sqrt(result.fun/len(P_data)/2))

# MAE
mae1_sum = 0
mae2_sum = 0
for i in range(len(P_data)):
    P_model = get_static_func(theta_data[i][0], theta_data[i][1], optimal_params)
    # cal MAE
    mae1 = np.mean(np.abs(P_data[i][0] - P_model[0]))
    mae2 = np.mean(np.abs(P_data[i][1] - P_model[1]))
    mae1_sum += mae1
    mae2_sum += mae2
print("回归得到的MAE1:", mae1_sum/len(P_data))
print("回归得到的MAE2:", mae2_sum/len(P_data))

print("回归得到的参数:", optimal_params)
# 刚度的两倍
print("回归得到的刚度:", optimal_params[0]*2, optimal_params[1]*2)

# 检查k的作用：在回归参数上加上噪声，看损失的变化
delta_k = 1
params_k = optimal_params.copy()
params_k[0] += delta_k
params_k[1] += delta_k
loss_k = objective_function(params_k, theta_data, P_data)
print(f"损失增加: {loss_k - result.fun}")
# 百分比增加
print(f"损失增加百分比: {(loss_k - result.fun)/result.fun*100}%")