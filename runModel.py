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
matplotlib.use('TkAgg')


path1 = "/Users/flypig/Documents/Coding/MujocoLearn/1101/1101/mydog110.xml"
path2 = '/Users/flypig/Documents/Coding/MujocoLearn/v2_4/urdf/dog2_4singleLeg.xml'
path3 = '/Users/flypig/Documents/Coding/MujocoLearn/v2_4/urdf/Exp_MAA.xml'
path4 = "/Users/flypig/Documents/Coding/MujocoLearn/v2_4/urdf/Exp_singleKMC.xml"

ExpName = "SingleLeg-Super"
m = mujoco.MjModel.from_xml_path(path2)
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

target_force_MAA_list = np.linspace(-20, 20, 161)
target_force_BAA_list = np.linspace(-20, 20, 161)

# with mujoco.viewer.launch_passive(m, d) as viewer:
for MAAPressure in target_force_MAA_list:
  for BAAPressure in target_force_BAA_list:
    # 3s 时间步长后关闭viewer
    print(f"Exp with MAA: {MAAPressure}, BAA: {BAAPressure}")
    start = d.time
    try:
      while d.time - start < 15: # viewer.is_running() and
        step += 1
        step_start = d.time
        # print(d.time)

        time_step = 5
        start_force_M = -20  # SOTA: -500
        target_force_MAA = MAAPressure
        target_force_BAA = BAAPressure
        # if step_start % time_step < time_step/2:
          # if flip == 0:
            # d.ctrl[:] = start_force_M+20    # 20N

        if d.time - start > 5 and d.time - start < 14: # 稳定时间
          d.ctrl[:2] = target_force_MAA    # 50N
          d.ctrl[2:] = target_force_BAA    # 50N
          record = True
          # flip = 1
        # else:
          # d.ctrl[:] = start_force_M   # 0N
          # d.ctrl[:] = start_force_M
          # flip = 0

        # 记录传感器数据
        # print(d.sensordata)
        # sensor_list.append(np.array([d.sensordata[0], d.sensordata[1], d.sensordata[2], d.sensordata[3]]))

        if d.time - start > 14 and record == True:
          StaticPositon = np.array(d.qpos)
          res = np.hstack(([target_force_MAA, target_force_BAA], StaticPositon))
          ExpResultList.append(res)
          record = False
        # print(d.qpos)
        # RobotState = np.array(d.qpos)
        # time_list.append(d.time)
        # sensor_list.append(RobotState)

        mujoco.mj_step(m, d)  # update!

        # 获取物理状态的更改，应用扰动，从GUI更新选项。
        # viewer.sync()   # TODO

        # 粗略的计时，相对于挂钟会有漂移。
        # time_until_next_step = m.opt.timestep - (time.time() - step_start)
        # if time_until_next_step > 0:
        #   time.sleep(time_until_next_step)
        # time.sleep(0.01)

    # # 按住ctrl C退出循环
    except KeyboardInterrupt:
      pass


print(ExpResultList)
ExpResultList = np.array(ExpResultList)
# 获取文件名
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ExpTime = current_time
folder_path = f"./data/Exp-{ExpName}-{ExpTime}"
os.makedirs(folder_path, exist_ok=False)  # 如果文件名称冲突，报错!
staticPlace_list_filename = os.path.join(folder_path, "StaticState_list.npy")
np.save(staticPlace_list_filename, ExpResultList)


if False:
  # 循环结束时，记录数据
  time_list = np.array(time_list)
  sensor_list = np.array(sensor_list)
  print("Time list shape:", time_list.shape)
  print("Sensor list:", sensor_list)
  # 当前路径下保存
  # 获取当前时间并格式化
  current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

  # 获取文件名

  ExpTime = current_time
  folder_path = f"./data/Exp-{ExpName}-{ExpTime}"
  os.makedirs(folder_path, exist_ok=False)  # 如果文件名称冲突，报错!

  # 生成文件名
  time_list_filename = os.path.join(folder_path, "time_list.npy")
  sensor_list_filename = os.path.join(folder_path, "sensor_list.npy")

  # 保存数据
  np.save(time_list_filename, time_list)
  np.save(sensor_list_filename, sensor_list)

  # 绘制图
  # plt.plot(time_list, sensor_list[:, 0])
  # plt.plot(time_list, sensor_list[:, 1])
  # print("Save!")
  # plt.savefig("./fig/sensor.png")
  # show the fig
  # plt.show()
  input("type to end")