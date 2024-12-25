import time

import mujoco
import mujoco.viewer
import numpy as np
import random
import math

path1 = "/Users/flypig/Documents/Coding/MujocoLearn/1101/1101/mydog110.xml"
path2 = '/Users/flypig/Documents/Coding/MujocoLearn/v2_4/urdf/dog2_4singleLeg.xml'
path3 = '/Users/flypig/Documents/Coding/MujocoLearn/v2_4/urdf/dog2_4.xml'
m = mujoco.MjModel.from_xml_path(path3)
d = mujoco.MjData(m)
flag = 0

# debug:打印质量
print(m.body_mass)

d.ctrl = np.zeros_like(d.ctrl)

mujoco.mj_step(m, d)

d.ctrl[:] = -0
# d.ctrl[4:8] = 0.4*100
# d.ctrl[8:] = 0
# d.ctrl[[1,3]] = 1

print()
input("Start")
# d.qpos[7:] = np.ones_like(d.ctrl) * np.pi/6
time_list = []
sensor_list = []

step = 0
flip = 0
with mujoco.viewer.launch_passive(m, d) as viewer:
  # 30时间步长后关闭viewer
  start = time.time()
  try:
    while viewer.is_running() and time.time() - start < 100:
      step += 1
      step_start = time.time()

      time_step = 5
      start_force_M = -0  # SOTA: -500
      start_force_B = -0  # SOTA: -200
      if time.time()%time_step < time_step/2:
        if flip == 0:
          # 随机给控制指令，但是保证0=1，2=3
          d.ctrl[:4] = start_force_M+1.962*2
          d.ctrl[4:] = start_force_B+15.042*2
          flip = 1
      else:
        d.ctrl[[0,1]] = start_force_M
        d.ctrl[[2,3]] = start_force_B
        flip = 0

      # 记录传感器数据
      # print(d.sensordata)
      # time_list.append(d.time)
      # sensor_list.append(np.array([d.sensordata[0], d.sensordata[1], d.sensordata[2], d.sensordata[3]]))
      # Mj_step可以替换为同样求值的代码
      # mj_step可以替换为同样评估策略并在执行物理之前应用控制信号的代码。

      # add control
      if flag == 0:
          flag = 1

      mujoco.mj_step(m, d)

      # print state
      print(d.qpos)
      # 查看器选项的修改示例：每两秒钟切换一次接触点。
      # with viewer.lock():
      #   viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

      # 获取物理状态的更改，应用扰动，从GUI更新选项。
      viewer.sync()

      # 粗略的计时，相对于挂钟会有漂移。
      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)

    # 循环结束时，记录数据
    time_list = np.array(time_list)
    sensor_list = np.array(sensor_list)
    print("Time list shape:", time_list.shape)
    print("Sensor list:", sensor_list)
    # 当前路径下保存
    np.save("./time_list.npy", time_list)
    np.save("./sensor_list.npy", sensor_list)
    print("Save!")
  # # 按住ctrl C退出循环
  except KeyboardInterrupt:
    pass
