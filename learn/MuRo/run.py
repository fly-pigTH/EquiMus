import time

import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path('/Users/flypig/Documents/Coding/MujocoLearn/mujoco_menagerie/unitree_go1/scene.xml')
d = mujoco.MjData(m)
flag = 0

d.ctrl = np.zeros(12)

# d.ctrl[[0,1,2,3]] = 0.2
# d.ctrl[4:8] = 0.4

d.ctrl[[2,5,8,11]] = 0
# d.ctrl[[1,3]] = 1

print()
input("Start")
# d.qpos[7:] = np.ones_like(d.ctrl) * np.pi/6

with mujoco.viewer.launch_passive(m, d) as viewer:
  # 30时间步长后关闭viewer
  start = time.time()
  while viewer.is_running() and time.time() - start < 1000:
    step_start = time.time()

    # Mj_step可以替换为同样求值的代码
    # mj_step可以替换为同样评估策略并在执行物理之前应用控制信号的代码。

    # add control
    if flag == 0:
        
        flag = 1

    # 给正弦信号（每隔四个）
    d.ctrl[2] = 1


    mujoco.mj_step(m, d)

    # print state
    print(d.qpos)

    # 查看器选项的修改示例：每两秒钟切换一次接触点。
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # 获取物理状态的更改，应用扰动，从GUI更新选项。
    viewer.sync()

    # 粗略的计时，相对于挂钟会有漂移。
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)