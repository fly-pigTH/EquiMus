# 仿真静力学平衡位置数据
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
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.auto_record import record_experiment
matplotlib.use('TkAgg')


path = "./models/v2_4/urdf/dog2_4singleLeg.xml"

ExpName = "Static_ideal_gridScan"
m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)
flag = 0

# debug:打印质量
print(m.body_mass)
time.sleep(0.8)

d.ctrl = np.zeros_like(d.ctrl)
d.ctrl[:] = 0

mujoco.mj_step(m, d)  # update!

time_list = []
sensor_list = []

step = 0
flip = 0

ExpResultList = []

# target_force_MAA_list = np.linspace(-20, 20, 161)
# target_force_BAA_list = np.linspace(-20, 20, 161)

# read the file
# MAAP = scipy.io.loadmat('F1.mat').get('F1')
# BAAP = scipy.io.loadmat('F2.mat').get('F2')

# Fixed
MAAP = scipy.io.loadmat('./src/simulation_verify/static/F1_10.mat').get("F1_matrix") # new data
BAAP = scipy.io.loadmat('./src/simulation_verify/static/F2_10.mat').get("F2_matrix")

# Landing
# MAAP = scipy.io.loadmat('./src/simulation_verify/static/F1_landing2DOF.mat').get("F1_matrix") # new data
# BAAP = scipy.io.loadmat('./src/simulation_verify/static/F2_landing2DOF.mat').get("F2_matrix") # new data

print(MAAP.shape)
# input()

print(f"MAAP: {MAAP}")
print(f"BAAP: {BAAP}")
input()


theta_1_array = np.linspace(math.pi/6, math.pi*2/3, 20) # 切分十个点
theta_2_array = np.linspace(0, math.pi/2, 20)

MAAP_list = MAAP
BAAP_list = BAAP

# Pre_list = [-4.71529817, 13.85488671]

# MAAP_list = [Pre_list[0]]
# BAAP_list = [Pre_list[1]]

# theta_1_array = [math.pi/2]
# theta_2_array = [math.pi/2]


print(MAAP[0][0], BAAP[0][0])
# input()

viewer_flag = 1

if viewer_flag:
  with mujoco.viewer.launch_passive(m, d) as viewer:
    for i in tqdm(range(len(theta_1_array))):
      for j in range(len(theta_2_array)):
        # 设置初始位置
        tar_theta1 = theta_1_array[i]
        tar_theta2 = theta_2_array[j]
        mujoco.mj_forward(m, d)

        MAAPressure = MAAP[i][j]  
        BAAPressure = BAAP[i][j]

        # MAAPressure = MAAP_list[i]  
        # BAAPressure = BAAP_list[j]
        # 3s 时间步长后关闭viewer
        # print(f"Exp {i}-{j} with MAA: {MAAPressure}, BAA: {BAAPressure}")
        start = time.time() if viewer_flag else d.time
        try:
          while (time.time() if viewer_flag else d.time) - start < 10: # viewer.is_running() and
            # print(time.time()-start if viewer_flag else d.time)
            time_now = time.time() if viewer_flag else d.time
            
            step += 1
            step_start = time.time() if viewer_flag else d.time

            time_step = 5
            target_force_MAA = MAAPressure
            target_force_BAA = BAAPressure
            
            if time_now - start > 0 and time_now - start < 8: # 稳定时间
              d.ctrl[:2] = target_force_MAA    # 50N
              d.ctrl[2:] = target_force_BAA    # 50N
              record = True

            if time_now - start >= 8 and record == True:
              StaticPositon = np.array(d.qpos)
              res = np.hstack(([target_force_MAA, target_force_BAA], StaticPositon))
              ExpResultList.append(res)
              # print(f"Pos: {d.qpos+math.pi/2}")

              # exp info
              print(f"Exp {i}-{j} with F1: {MAAPressure}, F2: {BAAPressure}")
              print(f"[Max Error]: {max(abs(tar_theta1 - d.qpos[0] - math.pi/2), abs(tar_theta2 - d.qpos[1] - math.pi/2))}, [Error of Tar1]:", tar_theta1 - d.qpos[0] - math.pi/2, "[Error of Tar2]:", tar_theta2 - d.qpos[1] - math.pi/2)
              record = False

            mujoco.mj_step(m, d)  # update!

            if viewer_flag:
              # 获取物理状态的更改，应用扰动，从GUI更新选项。
              viewer.sync()   # TODO
              # 粗略的计时，相对于挂钟会有漂移。
              time_until_next_step = m.opt.timestep - (time.time() - step_start)
              if time_until_next_step > 0:
                time.sleep(time_until_next_step)
              # time.sleep(0.01)

        # # 按住ctrl C退出循环
        except KeyboardInterrupt:
          raise KeyboardInterrupt

print(d.qpos)


print(ExpResultList)
ExpResultList = np.array(ExpResultList)

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
    "notes": "real 模型大实验，测试"
}

record_experiment(exp_config)
print("Exp Saved!")