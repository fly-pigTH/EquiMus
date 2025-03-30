# 用于对比实际实验数据

import time
import mujoco
import mujoco.viewer
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import mediapy as media
import datetime
import os
import pandas as pd



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

    m_3, m_4 = 0.18, 0.18

    # 刚度参数
    # k_3, k_4 = 300/2, 300/2
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

# path = "./models/SingleLeg_ideal_contrast.xml"
path = "./models/v2_4/urdf/dog2_4singleLeg_realconstrast.xml"
# dog2_4singleLeg_realconstrast.xml

ExpName = "sim-real constrast"
m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)
flag = 0

# Init the model
print(m.geom)
d.ctrl = np.zeros_like(d.ctrl)
mujoco.mj_step(m, d)  # update!

# Load the parameter and change the property of the model
s1 = 0.0003538647203395965
s2 = 0.00034663180121508557
k1 = 142.22034830974255
k2 = 129.49423449130097
l10 = 0.1641527577313371
l20 = 0.2579251802483194

# Experiment Settings
StartPoint = [math.pi/2, 0]
EndPoint = [math.pi/2, 0]

# Read the input-pressure file
data = pd.read_csv("./log/realdata/StaticProcess/real_StaticPoint_6group_2025-02-21_08-55-07.csv")
# 全部做，对比
# P1_array = np.linspace(0, 50, 6)
# P2_array = np.linspace(0, 50, 6)

# P1_array = data['P1'].values
# P2_array = data['P2'].values

# test the 10kPa
P1_array = [10]
P2_array = [10]

theta1_array = data['theta1'].values
theta2_array = data['theta2'].values
P1_array = np.array(P1_array) * 1000
P2_array = np.array(P2_array) * 1000
theta1_array = np.array(theta1_array) * math.pi / 180
theta2_array = np.array(theta2_array) * math.pi / 180

step = 0
ExpResultList = []
viewer_flag = True

exp_num = theta1_array.shape
theta_static = []

if viewer_flag:
  with mujoco.viewer.launch_passive(m, d) as viewer:
    input("Press to start!")
    # print(f"Exp {i}-{j} with MAA: {MAAPressure}, BAA: {BAAPressure}")
    try:
      for p1, p2 in zip(P1_array, P2_array):
        for i in range(1):
      # for p1 in P1_array:
      #   for p2 in P2_array:

          start = time.time() if viewer_flag else d.time
          torecord_flag = True
          while (time.time() if viewer_flag else d.time) - start < 10: # viewer.is_running() and
            # print(time.time()-start if viewer_flag else d.time)
            step += 1
            step_time = time.time() if viewer_flag else d.time

            # show the state
            # INFO
            interval = 0.01
            if time.time()%0.1 < interval:
                Positon = np.array(d.qpos)
                print("\033[H\033[J")  # 使用ANSI转义序列清除终端输出
                print(f"Pressure command: \n{[float(p1)/1000, float(p2)/1000]} kPa")
                print(f"Theta_1: {(Positon[0]+math.pi/2)*180/math.pi:.2f}°")
                print(f"Theta_2: {(Positon[1]+math.pi/2)*180/math.pi:.2f}°")

            if step_time - start > 0 and step_time - start < 9: # 稳定时间
              F1 = p1*s1
              F2 = p2*s2
              d.ctrl[:2] = F1
              d.ctrl[2:] = F2

            if 9 < step_time - start < 10: # 稳定时间
              if torecord_flag:
                Positon = np.array(d.qpos)
                res = np.hstack(([F1, F2, d.time], Positon))
                ExpResultList.append(res)
                torecord_flag = False

            mujoco.mj_step(m, d)  # update!

            if viewer_flag:
              # 获取物理状态的更改，应用扰动，从GUI更新选项。
              viewer.sync()   # TODO
              # 粗略的计时，相对于挂钟会有漂移。
              time_until_next_step = m.opt.timestep - (time.time() - step_time)
              if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # # 按住ctrl C退出循环
    except KeyboardInterrupt:
      pass

ExpResultList = np.array(ExpResultList)
print(ExpResultList.shape)
print(ExpResultList)

# plt.plot(ExpResultList[:, 3])
input()

# 获取文件名
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ExpTime = current_time
folder_path = f"./data/real_contrast_static/Exp-{ExpName}-{ExpTime}"
os.makedirs(folder_path, exist_ok=False)  # 如果文件名称冲突，报错!
staticPlace_list_filename = os.path.join(folder_path, "StaticState_list.npy")
np.save(staticPlace_list_filename, ExpResultList)

