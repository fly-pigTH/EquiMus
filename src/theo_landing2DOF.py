# landing model 2DOF 静力学推导分析，用于计算初始qpos和对应的平衡力
# 静力学稳定性分析

import numpy as np
import math
import scipy.io
from qpos_cal import qpos_cal
import matplotlib.pyplot as plt
import time


# 返回静态力和标准qpos
def get_static_func_and_qpos_forLanding2DOF(theta_1, theta_2, mode="Landing2DOF"):
    '''
        使用标准theta1 和 theta2
        mode = Landing2DOF, Fixed
    '''
    a_1 = 0.25
    a_2 = 0.25
    b_1 = 0.21213
    b_2 = 0.1
    d_1 = 0.06
    d_2 = 0.10
    s = 0

    M = 0.155  # 真实值
    m_1, m_2 = 0.086, 0.1033
    I_1, I_2 = (1/12 * m_1 * 0.25**2), (1/12 * m_2 * 0.25**2)

    m_3, m_4 = 0.18, 0.18

    # 刚度参数
    k_3, k_4 = 637.52/2, 631.6/2

    # 阻尼参数
    c_3, c_4 = 22.68/2, 21.8/2

    # 重力加速度
    g = 9.8

    # 角度参数 (度转弧度)
    beta_1 = 8.13 / 180 * np.pi
    beta_2 = 30 / 180 * np.pi

    # 长度参数
    l_10 = 0.174
    l_20 = 0.252

    O_x_O = 0
    O_y_O = 0

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
    dO_x_O_dtheta_1 = 0
    dO_y_O_dtheta_1 = 0

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

    dO_x_O_dtheta_2 = 0
    dO_y_O_dtheta_2 = 0

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


    if mode == "Fixed":
        X_bias = 0
        Y_bias = 0

        dX_bias_dtheta_1 = 0
        dY_bias_dtheta_1 = 0
        dX_bias_dtheta_2 = 0
        dY_bias_dtheta_2 = 0
    else:
        X_bias = F_x_O
        Y_bias = F_y_O

        dX_bias_dtheta_1 = dF_x_O_dtheta_1
        dY_bias_dtheta_1 = dF_y_O_dtheta_1
        dX_bias_dtheta_2 = dF_x_O_dtheta_2
        dY_bias_dtheta_2 = dF_y_O_dtheta_2

    # 把以上所有参数都减掉F
    O_x_O = O_x_O - X_bias
    O_y_O = O_y_O - Y_bias
    A_x_O = A_x_O - X_bias
    A_y_O = A_y_O - Y_bias
    B_x_O = B_x_O - X_bias
    B_y_O = B_y_O - Y_bias
    C_x_O = C_x_O - X_bias
    C_y_O = C_y_O - Y_bias
    D_x_O = D_x_O - X_bias
    D_y_O = D_y_O - Y_bias
    E_x_O = E_x_O - X_bias
    E_y_O = E_y_O - Y_bias

    F_x_O -= X_bias
    F_y_O -= Y_bias

    # O 点
    dO_x_O_dtheta_1 -= dX_bias_dtheta_1
    dO_y_O_dtheta_1 -= dY_bias_dtheta_1
    dO_x_O_dtheta_2 -= dX_bias_dtheta_2
    dO_y_O_dtheta_2 -= dY_bias_dtheta_2

    # A 点
    dA_x_O_dtheta_1 -= dX_bias_dtheta_1
    dA_y_O_dtheta_1 -= dY_bias_dtheta_1
    dA_x_O_dtheta_2 -= dX_bias_dtheta_2
    dA_y_O_dtheta_2 -= dY_bias_dtheta_2

    # B 点
    dB_x_O_dtheta_1 -= dX_bias_dtheta_1
    dB_y_O_dtheta_1 -= dY_bias_dtheta_1
    dB_x_O_dtheta_2 -= dX_bias_dtheta_2
    dB_y_O_dtheta_2 -= dY_bias_dtheta_2

    # C 点
    dC_x_O_dtheta_1 -= dX_bias_dtheta_1
    dC_y_O_dtheta_1 -= dY_bias_dtheta_1
    dC_x_O_dtheta_2 -= dX_bias_dtheta_2
    dC_y_O_dtheta_2 -= dY_bias_dtheta_2


    # D 点
    dD_x_O_dtheta_1 -= dX_bias_dtheta_1
    dD_y_O_dtheta_1 -= dY_bias_dtheta_1
    dD_x_O_dtheta_2 -= dX_bias_dtheta_2
    dD_y_O_dtheta_2 -= dY_bias_dtheta_2

    # E 点
    dE_x_O_dtheta_1 -= dX_bias_dtheta_1
    dE_y_O_dtheta_1 -= dY_bias_dtheta_1
    dE_x_O_dtheta_2 -= dX_bias_dtheta_2
    dE_y_O_dtheta_2 -= dY_bias_dtheta_2

    # F
    dF_x_O_dtheta_1 -= dX_bias_dtheta_1
    dF_y_O_dtheta_1 -= dY_bias_dtheta_1
    dF_x_O_dtheta_2 -= dX_bias_dtheta_2
    dF_y_O_dtheta_2 -= dY_bias_dtheta_2

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

    # 等式右侧
    RHSb_1 = -( m_1*g*((dE_y_O_dtheta_1+dO_y_O_dtheta_1)/2) + m_2*g*((dE_y_O_dtheta_1+dF_y_O_dtheta_1)/2) + m_3*g*((dA_y_O_dtheta_1+dC_y_O_dtheta_1)/2) + m_4*g*((dB_y_O_dtheta_1+dD_y_O_dtheta_1)/2) + M*g*dO_y_O_dtheta_1)
    RHSb_2 = -( m_1*g*((dE_y_O_dtheta_2+dO_y_O_dtheta_2)/2) + m_2*g*((dE_y_O_dtheta_2+dF_y_O_dtheta_2)/2) + m_3*g*((dA_y_O_dtheta_2+dC_y_O_dtheta_2)/2) + m_4*g*((dB_y_O_dtheta_2+dD_y_O_dtheta_2)/2) + M*g*dO_y_O_dtheta_2)
    b = np.array([RHSb_1, RHSb_2])

    # 等式左侧
    LHSA = np.array([[dl_1_dtheta_1, dl_2_dtheta_1], 
                    [dl_1_dtheta_2, dl_2_dtheta_2]])

    # 回复力
    F_k = np.array([k_3*(l_1-l_10), k_4*(l_2-l_20)])

    # 计算静力学
    StaticForce = np.linalg.solve(LHSA, b) + F_k

    qpos = np.zeros(10)
    qpos[0] = -O_x_O
    qpos[1] = -(O_y_O + 2*a_1)
    qpos[2:4] = [theta_1-math.pi/2, theta_2]
    qpos_cal_res = qpos_cal(theta_1, theta_2)
    qpos_cal_res_0 = qpos_cal(math.pi/2, 0)
    l1_start = qpos_cal_res_0[4]
    l2_start = qpos_cal_res_0[5]
    uang_1_start = qpos_cal_res_0[2]
    uang_2_start = qpos_cal_res_0[3]
    qpos[4] = qpos_cal_res[2]-uang_1_start
    qpos[7] = qpos_cal_res[3]-uang_2_start
    qpos[5:7] = (qpos_cal_res[4:5]-l1_start)/2   # l_1, l_2
    qpos[8:10] = (qpos_cal_res[5:6]-l2_start)/2   # l_1, l_2
    return StaticForce, qpos

# 势能分析
def static_energy(theta_1, theta_2, StaticForce, mode="Landing2DOF"):
    '''
        使用标准theta1 和 theta2, 绘制在StaticForce作用下-theta_1, theta_2处的势能
    '''
    a_1 = 0.25
    a_2 = 0.25
    b_1 = 0.21213
    b_2 = 0.1
    d_1 = 0.06
    d_2 = 0.10
    s = 0

    M = 0.155  # 真实值
    m_1, m_2 = 0.086, 0.1033
    I_1, I_2 = (1/12 * m_1 * 0.25**2), (1/12 * m_2 * 0.25**2)

    m_3, m_4 = 0.18, 0.18

    # 刚度参数
    k_3, k_4 = 637.52/2, 631.6/2

    # 阻尼参数
    c_3, c_4 = 22.68/2, 21.8/2

    # 重力加速度
    g = 9.8

    # 角度参数 (度转弧度)
    beta_1 = 8.13 / 180 * np.pi
    beta_2 = 30 / 180 * np.pi

    # 长度参数
    l_10 = 0.174
    l_20 = 0.252

    O_x_O = 0
    O_y_O = 0

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
    dO_x_O_dtheta_1 = 0
    dO_y_O_dtheta_1 = 0

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

    dO_x_O_dtheta_2 = 0
    dO_y_O_dtheta_2 = 0

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

    if mode == "Fixed":
        X_bias = 0
        Y_bias = 0

        dX_bias_dtheta_1 = 0
        dY_bias_dtheta_1 = 0
        dX_bias_dtheta_2 = 0
        dY_bias_dtheta_2 = 0
    else:
        X_bias = F_x_O
        Y_bias = F_y_O

        dX_bias_dtheta_1 = dF_x_O_dtheta_1
        dY_bias_dtheta_1 = dF_y_O_dtheta_1
        dX_bias_dtheta_2 = dF_x_O_dtheta_2
        dY_bias_dtheta_2 = dF_y_O_dtheta_2

    # 把以上所有参数都减掉F
    O_x_O = O_x_O - X_bias
    O_y_O = O_y_O - Y_bias
    A_x_O = A_x_O - X_bias
    A_y_O = A_y_O - Y_bias
    B_x_O = B_x_O - X_bias
    B_y_O = B_y_O - Y_bias
    C_x_O = C_x_O - X_bias
    C_y_O = C_y_O - Y_bias
    D_x_O = D_x_O - X_bias
    D_y_O = D_y_O - Y_bias
    E_x_O = E_x_O - X_bias
    E_y_O = E_y_O - Y_bias

    F_x_O -= X_bias
    F_y_O -= Y_bias

    # O 点
    dO_x_O_dtheta_1 -= dX_bias_dtheta_1
    dO_y_O_dtheta_1 -= dY_bias_dtheta_1
    dO_x_O_dtheta_2 -= dX_bias_dtheta_2
    dO_y_O_dtheta_2 -= dY_bias_dtheta_2

    # A 点
    dA_x_O_dtheta_1 -= dX_bias_dtheta_1
    dA_y_O_dtheta_1 -= dY_bias_dtheta_1
    dA_x_O_dtheta_2 -= dX_bias_dtheta_2
    dA_y_O_dtheta_2 -= dY_bias_dtheta_2

    # B 点
    dB_x_O_dtheta_1 -= dX_bias_dtheta_1
    dB_y_O_dtheta_1 -= dY_bias_dtheta_1
    dB_x_O_dtheta_2 -= dX_bias_dtheta_2
    dB_y_O_dtheta_2 -= dY_bias_dtheta_2

    # C 点
    dC_x_O_dtheta_1 -= dX_bias_dtheta_1
    dC_y_O_dtheta_1 -= dY_bias_dtheta_1
    dC_x_O_dtheta_2 -= dX_bias_dtheta_2
    dC_y_O_dtheta_2 -= dY_bias_dtheta_2


    # D 点
    dD_x_O_dtheta_1 -= dX_bias_dtheta_1
    dD_y_O_dtheta_1 -= dY_bias_dtheta_1
    dD_x_O_dtheta_2 -= dX_bias_dtheta_2
    dD_y_O_dtheta_2 -= dY_bias_dtheta_2

    # E 点
    dE_x_O_dtheta_1 -= dX_bias_dtheta_1
    dE_y_O_dtheta_1 -= dY_bias_dtheta_1
    dE_x_O_dtheta_2 -= dX_bias_dtheta_2
    dE_y_O_dtheta_2 -= dY_bias_dtheta_2

    # F
    dF_x_O_dtheta_1 -= dX_bias_dtheta_1
    dF_y_O_dtheta_1 -= dY_bias_dtheta_1
    dF_x_O_dtheta_2 -= dX_bias_dtheta_2
    dF_y_O_dtheta_2 -= dY_bias_dtheta_2

    # 计算长度 l1 和 l2
    l_1 = np.sqrt((A_x_O - C_x_O)**2 + (A_y_O - C_y_O)**2)
    l_2 = np.sqrt((B_x_O - D_x_O)**2 + (B_y_O - D_y_O)**2)

    # 势能
    E_PG = -(m_1*g*(O_y_O + E_y_O)/2 + m_2*g*(E_y_O + F_y_O)/2 + m_3*g*(C_y_O + A_y_O)/2 + m_4*g*(D_y_O + B_y_O)/2 + M*g*O_y_O)
    E_PK = 1/2*k_3*(l_1-l_10)**2 + 1/2*k_4*(l_2-l_20)**2
    E_F = -np.dot(StaticForce, [l_1-l_10, l_2-l_20])
    return E_PG + E_PK + E_F


# 应用参数：绘制某个F_1, F_2处的能量图（力坐标）
def draw_energy_in_force(StaticForce, mode):
    # plot the potential energy
    theta_1 = np.linspace(0, math.pi*2/3, 20)
    theta_2 = np.linspace(0, math.pi*2/3, 20)

    E = np.zeros((len(theta_1), len(theta_2)))
    for i in range(len(theta_1)):
        for j in range(len(theta_2)):
            E[i, j] = static_energy(theta_1[i], theta_2[j], StaticForce, mode=mode)

    # 为了和MeshGrid统一，这里需要转置，使X对应横坐标
    E = E.T

    X, Y = np.meshgrid(theta_1, theta_2)
    
    # 填充等高线
    contour_filled = plt.contourf(X, Y, E, levels=200, cmap='viridis')
    plt.colorbar(contour_filled, label="Value")  # 添加颜色条

    # 绘制等高线
    contour_lines = plt.contour(X, Y, E, levels=200, colors='black', linewidths=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8)  # 添加线上的标签

    # test
    plt.scatter(-0.3445+math.pi/2, 1.04)
    plt.scatter(-0.0279+math.pi/2, 0.251)
    center_energy = static_energy(-0.0279+math.pi/2, 0.251, StaticForce, mode=mode)

    # 设置标题和轴标签
    plt.title("Contour Plot")
    plt.xlabel("Theta1 Coordinate")
    plt.ylabel("Theta2 Coordinate")
    # plt.title(f'Potential Energy with \nsStatic Point at theta_1 = {theta_1}.2f, theta_2 = {theta_2}.2f')
    plt.title(f"StaticForce={StaticForce}, Center Energy={center_energy}")
    plt.show()
    pass

# 应用参数：绘制某个theta_1, theta_2处的能量图（角度坐标）
def draw_energy_in_angle(theta_10, theta_20, mode):
    '''
        绘制在theta_10, theta_20处的势能图
    '''
    StaticForce, qpos = get_static_func_and_qpos_forLanding2DOF(theta_10, theta_20, mode=mode)
    print(f"theta_10 = {theta_10}, theta_20 = {theta_20}, StaticForce = {StaticForce}, qpos = {qpos}")

    # plot the potential energy
    theta_1 = np.linspace(0, math.pi*2/3, 50)
    theta_2 = np.linspace(0, math.pi*2/3, 50)

    E = np.zeros((len(theta_1), len(theta_2)))
    for i in range(len(theta_1)):
        for j in range(len(theta_2)):
            E[i, j] = static_energy(theta_1[i], theta_2[j], StaticForce, mode=mode)

    # 为了和MeshGrid统一，这里需要转置，使X对应横坐标
    E = E.T

    X, Y = np.meshgrid(theta_1, theta_2)

    # 填充等高线
    contour_filled = plt.contourf(X, Y, E, levels=200, cmap='viridis')
    plt.colorbar(contour_filled, label="Value")  # 添加颜色条

    # 绘制等高线
    contour_lines = plt.contour(X, Y, E, levels=200, colors='black', linewidths=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8)  # 添加线上的标签

    plt.scatter(theta_10, theta_20, label="Static Point", marker='o', color='red')  # 绘制静态点

    # 设置标题和轴标签
    plt.title("Contour Plot")
    plt.xlabel("Theta1 Coordinate")
    plt.ylabel("Theta2 Coordinate")
    plt.show()

# draw_energy_in_force(np.array([15.2, 24.8]), "Landing2DOF")
# theta_0 = [-0.0279+math.pi/2, 0.251]
# theta_0 = [76*math.pi/180, 38*math.pi/180]
# print(f"Theta_0 = {theta_0}")
# draw_energy_in_angle(*theta_0, mode="Landing2DOF")


if __name__ == '__main__':
    
    runMode = "Landing2DOF"

    plt.figure(figsize=(10,8))

    for theta_10 in np.linspace(0, math.pi*2/3, 20):
        for theta_20 in np.linspace(0, math.pi*2/3, 20):
            StaticForce, qpos = get_static_func_and_qpos_forLanding2DOF(theta_10, theta_20, mode=runMode)
            print(f"theta_1 = {theta_10}, theta_2 = {theta_20}, StaticForce = {StaticForce}, qpos = {qpos}")

            # plot the potential energy
            theta_1 = np.linspace(0, math.pi*2/3, 20)
            theta_2 = np.linspace(0, math.pi*2/3, 20)
            print(f"Static Energy = {static_energy(theta_10, theta_20, StaticForce, runMode)}")

            E = np.zeros((len(theta_1), len(theta_2)))
            for i in range(len(theta_1)):
                for j in range(len(theta_2)):
                    E[i, j] = static_energy(theta_1[i], theta_2[j], StaticForce, runMode)

            # 为了和MeshGrid统一，这里需要转置，使X对应横坐标
            E = E.T

            X, Y = np.meshgrid(theta_1, theta_2)
            
            # 填充等高线
            print("All Shape:", X.shape, Y.shape, E.shape)
            if False:   # 角度图
                contour_filled = plt.contourf(X, Y, E, levels=40, cmap='viridis')
                # plt.colorbar(contour_filled, label="Value")  # 添加颜色条

                # 绘制等高线
                contour_lines = plt.contour(X, Y, E, levels=40, colors='black', linewidths=0.5)
                plt.clabel(contour_lines, inline=True, fontsize=8)  # 添加线上的标签

                plt.plot(theta_10, theta_20, label="Static Point", marker='o', color='red')  # 绘制静态点

                # 设置标题和轴标签
                plt.title("Contour Plot")
                plt.xlabel("Theta1 Coordinate")
                plt.ylabel("Theta2 Coordinate")
                plt.title('Potential Energy with \nsStatic Point at theta_1 = %.2f, theta_2 = %.2f' % (theta_10, theta_20))

            # 显示图像

            # plt.show(block=False)
            # plt.pause(0.1)
            # plt.cla()

            # 绘制机器人足端坐标
            End_x = []
            End_y = []
            Mid_x = []
            Mid_y = []
            Base_x = []     # for Landing2DOF
            Base_y = []
            for i in range(len(theta_1)):
                for j in range(len(theta_2)):
                    end_x1 = 0.25 * np.cos(theta_1[i]) + 0.25 * np.cos(theta_1[i] + theta_2[j])
                    end_y1 = 0.25 * np.sin(theta_1[i]) + 0.25 * np.sin(theta_1[i] + theta_2[j])
                    base_x1 = -end_x1
                    base_y1 = -end_y1
                    Mid_x.append(0.25 * np.cos(theta_1[i]))
                    Mid_y.append(0.25 * np.sin(theta_1[i]))
                    End_x.append(end_x1)
                    End_y.append(end_y1)
                    Base_x.append(base_x1)
                    Base_y.append(base_y1)
            End_x = np.array(End_x).reshape(len(theta_1), len(theta_2))
            End_y = np.array(End_y).reshape(len(theta_1), len(theta_2))
            Mid_x = np.array(Mid_x).reshape(len(theta_1), len(theta_2))
            Mid_y = np.array(Mid_y).reshape(len(theta_1), len(theta_2))
            
            # 全部转置，对应meshgrid
            End_x = End_x.T
            End_y = End_y.T
            Mid_x = Mid_x.T
            Mid_y = Mid_y.T
            Base_x = np.array(Base_x).reshape(len(theta_1), len(theta_2)).T
            Base_y = np.array(Base_y).reshape(len(theta_1), len(theta_2)).T

            print(End_x.shape, End_y.shape)

            if runMode == "Landing2DOF":
                contour_filled = plt.contourf(Base_x, Base_y, E, levels=20, cmap='viridis', linewidths=0.5)
                contour_lines = plt.contour(Base_x, Base_y, E, levels=20, colors='black', linewidths=0.5)
                plt.clabel(contour_lines, inline=True, fontsize=8)  # 添加线上的标签
            else:
                contour_filled = plt.contourf(End_x, End_y, E, levels=20, cmap='viridis', linewidths=0.5)
                contour_lines = plt.contour(End_x, End_y, E, levels=20, colors='black', linewidths=0.5)
                plt.clabel(contour_lines, inline=True, fontsize=8)  # 添加线上的标签
            plt.xlabel('x')
            plt.ylabel('y')

            # 绘制足端位置
            origin_x = 0    # 坐标原点
            origin_y = 0
            if runMode == "Landing2DOF":
                origin_x = 0.25 * np.cos(theta_10) + 0.25 * np.cos(theta_10 + theta_20)
                origin_y = 0.25 * np.sin(theta_10) + 0.25 * np.sin(theta_10 + theta_20)
            else:
                origin_x = 0
                origin_y = 0
            O_array_x = 0 - origin_x
            O_array_y = 0 - origin_y
            E_array_x = 0.25 * np.cos(theta_10) - origin_x
            E_array_y = 0.25 * np.sin(theta_10) - origin_y
            F_array_x = 0.25 * np.cos(theta_10) + 0.25 * np.cos(theta_10 + theta_20) - origin_x
            F_array_y = 0.25 * np.sin(theta_10) + 0.25 * np.sin(theta_10 + theta_20) - origin_y

            plt.plot(O_array_x, O_array_y, marker='o', color='red', label='O')
            plt.plot(E_array_x, E_array_y, marker='o', color='blue', label='E')
            plt.plot(F_array_x, F_array_y, marker='o', color='blue', label='F')
            plt.plot([O_array_x, E_array_x, F_array_x], [O_array_y, E_array_y, F_array_y], color='green', linestyle='--')
            
            # 获取当前轴对象并倒立坐标轴
            current_axes = plt.gca()
            current_axes.set_xlim(-0.5, 0.5)
            current_axes.set_ylim(-0.5, 0.5)
            current_axes.invert_xaxis()  # 倒立X轴
            current_axes.invert_yaxis()  # 倒立Y轴
            current_axes.set_aspect(1)
            plt.grid(linestyle='--', linewidth=0.5)
            plt.legend()
            plt.title(f"runMode: {runMode}\n theta10: {round(theta_10*180/math.pi, 4)}deg, theta20: {round(theta_20*180/math.pi, 4)}deg")

            # 循环播放
            plt.show(block=False)
            plt.pause(0.1)
            plt.cla()
            

