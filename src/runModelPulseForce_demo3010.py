
import time
import mujoco
import mujoco.viewer
import numpy as np
import random
import math
import mediapy as media
import datetime

path = "./models/v2_4/urdf/dog2_4singleLeg_realconstrast.xml"

s1 = 0.0003538647203395965
s2 = 0.00034663180121508557
damping_MAA, damping_BAA = 291.5757670770301/20, 98.3306672255234/8

m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)

RB_BAA_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_BAA_SlideJoint")
RB_BAA_FM_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_BAA_FM_SlideJoint")
RB_MAA_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_MAA_SlideJoint")
RB_MAA_FM_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_MAA_FM_SlideJoint")

m.dof_damping[RB_BAA_SlideJoint_id] = damping_BAA  # N·s/m
m.dof_damping[RB_BAA_FM_SlideJoint_id] = damping_BAA  # N·s/m
m.dof_damping[RB_MAA_SlideJoint_id] = damping_MAA  # N·s/m
m.dof_damping[RB_MAA_FM_SlideJoint_id] = damping_MAA  # N·s/m

# Init the model
print(m.body_mass)
d.ctrl = np.zeros_like(d.ctrl)
mujoco.mj_step(m, d)  # update!

ExpResultList = []
viewer_flag = True



if viewer_flag:
  with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time() if viewer_flag else d.time
    try:
      while (time.time() if viewer_flag else d.time) - start < 20: # viewer.is_running() and
        # print(time.time()-start if viewer_flag else d.time)
        step_time = time.time() if viewer_flag else d.time

        if step_time - start > 0 and step_time - start < 10: # 稳定时间
          d.ctrl[:2] = 30*1e3*s1
          d.ctrl[2:] = 10*1e3*s2

        if 10 <= step_time - start < 20: # 稳定时间
          d.ctrl[:2] = 0
          d.ctrl[2:] = 0
          Positon = np.array(d.qpos)

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