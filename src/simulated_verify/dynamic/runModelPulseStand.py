# Landing Model 动力学测试, 从初始加载直接施加[10, 20]N的力，保存数据

import time
import mujoco
import mujoco.viewer
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib
import mediapy as media
import datetime
import os
import pandas as pd
import scipy.io
from theo_landing2DOF import get_static_func_and_qpos_forLanding2DOF
import scipy.io
from tqdm import tqdm
from utils.auto_record import record_experiment

matplotlib.use('TkAgg')


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

path = "./models/SingleLeg_ideal_landing2DOF.xml"


ExpName = "landing 10-20N Force Pulse"
m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)
flag = 0

# Init the model
print(m.body_mass)
d.ctrl = np.zeros_like(d.ctrl)
mujoco.mj_step(m, d)  # update!

# Experiment Settings
StartPoint = [math.pi/2, 0]
EndPoint = [math.pi/2, 0]

# 加载stand实验数据
MAAP = scipy.io.loadmat('./data/matlab/F1_array_landing2DOF_stand.mat').get("F1_matrix").flatten() # new data
BAAP = scipy.io.loadmat('./data/matlab/F2_array_landing2DOF_stand.mat').get("F2_matrix").flatten()
print(MAAP.shape)


# Fstatic_end = get_static_func(EndPoint[0], EndPoint[1])
ForcePulse = [10, 20]

step = 0
ExpResultList = []
viewer_flag = False

init_force, init_qpos = get_static_func_and_qpos_forLanding2DOF(math.pi/180*75, math.pi/180*30)

# if viewer_flag:
#   with mujoco.viewer.launch_passive(m, d) as viewer:
start = time.time() if viewer_flag else d.time
try:
  while (time.time() if viewer_flag else d.time) - start < 30: # viewer.is_running() and
    # print(time.time()-start if viewer_flag else d.time)
    step += 1
    step_time = time.time() if viewer_flag else d.time

    if step_time - start > 0 and step_time - start < 10: # 稳定时间
      d.ctrl[:2] = ForcePulse[0]
      d.ctrl[2:] = ForcePulse[1]
      
      # d.qpos[0] = 0 # 强制设定位置！
      # d.qpos[3] = 0 # 强制设定位置！
      # d.qpos[2] = 0
      # mujoco.mju_copy(d.qpos, init_qpos)
      # d.ctrl[:2] = init_force[0]
      # d.ctrl[2:] = init_force[1]
      # d.ctrl[:2] = MAAP[0]
      # d.ctrl[2:] = BAAP[0]

    if 10 < step_time - start < 80: # 稳定时间
      rank = int((step_time - start)//1)

      print(f"rank={rank}")
      # d.qpos[2] = 0

      d.ctrl[:2] = ForcePulse[0]
      d.ctrl[2:] = ForcePulse[1]
      # d.ctrl[:2] = MAAP[rank]
      # d.ctrl[2:] = BAAP[rank]

    Positon = np.array(d.qpos)
    res = np.hstack(([ForcePulse[0], ForcePulse[1], d.time], Positon))
    ExpResultList.append(res)

    mujoco.mj_step(m, d)  # update!

    if viewer_flag:
      # 获取物理状态的更改，应用扰动，从GUI更新选项。
      # viewer.sync()   # TODO
      # 粗略的计时，相对于挂钟会有漂移。
      time_until_next_step = m.opt.timestep - (time.time() - step_time)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)

# # 按住ctrl C退出循环
except KeyboardInterrupt:
  pass

ExpResultList = np.array(ExpResultList)
print(ExpResultList.shape)

# 获取文件名
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ExpTime = current_time
folder_path = f"./data/Exp-{ExpName}-{ExpTime}"
os.makedirs(folder_path, exist_ok=False)  # 如果文件名称冲突，报错!
staticPlace_list_filename = os.path.join(folder_path, "StaticState_list.npy")
np.save(staticPlace_list_filename, ExpResultList)

# if success, record it in the Whole CSV
exp_config = {
    "id": f"{ExpName}-{ExpTime}",
    "start_time": datetime.datetime.now(),
    "dataFileName": staticPlace_list_filename,
    "notes": " Landing2DOF 单点 Model 动力学"
}

record_experiment(exp_config)