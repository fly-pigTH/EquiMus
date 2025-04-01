# Generate the swing trajectory for the simulated robot, plan scan (10, 10) x (10, 10) grid
import numpy as np
import math
# import matplotlib.pyplot as plt
# import mediapy as media
# import pandas as pd
from tqdm import tqdm
import rootpath
import sys
sys.path.append(rootpath.detect())
from utils.experiment import MujocoExperiment
import json

# 静力学计算模块
def get_static_func(theta_1, theta_2):

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

    # use measured value!
    m_3, m_4 = 0.18648, 0.27266
    k_3, k_4 = 637.52/2, 631.6/2
    l_10, l_20 = 0.174, 0.252

    # 阻尼参数
    c_3, c_4 = 22.68/2, 21.8/2

    # 重力加速度
    g = 9.8

    # 角度参数 (度转弧度)
    beta_1 = 8.13 / 180 * np.pi
    beta_2 = 30 / 180 * np.pi

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

    G_x_O = b_1 * np.cos(beta_1) * np.cos(theta_1)
    G_y_O = b_1 * np.cos(beta_1) * np.sin(theta_1)

    H_x_O = a_1 * np.cos(theta_1) + b_2 * np.cos(beta_2) * np.cos(theta_1 + theta_2)
    H_y_O = a_1 * np.sin(theta_1) + b_2 * np.cos(beta_2) * np.sin(theta_1 + theta_2)

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

    dG_x_O_dtheta_1 = -b_1 * np.cos(beta_1) * np.sin(theta_1)
    dG_y_O_dtheta_1 = b_1 * np.cos(beta_1) * np.cos(theta_1)

    dH_x_O_dtheta_1 = -a_1 * np.sin(theta_1) - b_2 * np.cos(beta_2) * np.sin(theta_1 + theta_2)
    dH_y_O_dtheta_1 = a_1 * np.cos(theta_1) + b_2 * np.cos(beta_2) * np.cos(theta_1 + theta_2)

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

    dH_x_O_dtheta_2 = -b_2 * np.cos(beta_2) * np.sin(theta_1 + theta_2)
    dH_y_O_dtheta_2 = b_2 * np.cos(beta_2) * np.cos(theta_1 + theta_2)


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
    return StaticForce

path = "./models/v2_4/urdf/dog2_4singleLeg_realconstrast.xml"
experiment_instance = MujocoExperiment(path)
fixed_para = {
    'stiffness_MAA': 318.76, #637.52 / 2,
    'stiffness_BAA': 315.8, #631.6 / 2,
    'l10': 0.174,
    'l20': 0.252,
    'damping_MAA': 11.34,
    'damping_BAA': 10.90,
    'c1_thigh': 0,
    'c2_calf': 0,
    's1': 1.0,
    's2': 1.0,
    'P1': 0,        # To be set
    'P2': 0,
    'P1_prime': 0.0,      
    'P2_prime': 0.0       # equal to F when s=1
}

theta1_array = np.linspace(math.pi/6, math.pi*2/3, 11)[1:]
theta2_array = np.linspace(0, math.pi/2, 11)[1:]
exp_data_serializable = []

for theta_1_start in tqdm(theta1_array, desc="Theta 1 Start Loop"):
    for theta_2_start in tqdm(theta2_array, desc="Theta 2 Start Loop", leave=False):
        for theta_1_end in tqdm(theta1_array, desc="Theta 1 End Loop", leave=False):
            for theta_2_end in tqdm(theta2_array, desc="Theta 2 End Loop", leave=False):
                fixed_para['P1'] = get_static_func(theta_1_start, theta_2_start)[0]
                fixed_para['P2'] = get_static_func(theta_1_start, theta_2_start)[1]
                fixed_para['P1_prime'] = get_static_func(theta_1_end, theta_2_end)[0]
                fixed_para['P2_prime'] = get_static_func(theta_1_end, theta_2_end)[1]
                time_sim_, theta1_sim_, theta2_sim_, frames_, valid_, valid_last_ = experiment_instance.run(fixed_para, 10, 20, False)
                exp_data_serializable.append({
                    'theta_1_start': theta_1_start,
                    'theta_2_start': theta_2_start,
                    'theta_1_end': theta_1_end,
                    'theta_2_end': theta_2_end,
                    'F1': fixed_para['P1'],
                    'F2': fixed_para['P2'],
                    'F1_prime': fixed_para['P1_prime'],
                    'F2_prime': fixed_para['P2_prime'],
                    'time_sim': time_sim_.tolist(),
                    'theta1_sim': theta1_sim_.tolist(),
                    'theta2_sim': theta2_sim_.tolist(),
                    'valid': valid_,
                    'valid_last': valid_last_
                })

# save as json file format
# 保存为 JSON 文件
with open("src/simulated_verify/dynamic/data/sqing_experiment_data.json", "w") as f:
    json.dump(exp_data_serializable, f, indent=4)
