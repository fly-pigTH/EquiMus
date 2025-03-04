
import time
import mujoco
import mujoco.viewer
import numpy as np
import random
import math
import mediapy as media
import datetime
# from runModel_static_realconstrast import bias_calculator

path = "./models/v2_4/urdf/dog2_4singleLeg_realconstrast.xml"

# Tools
# Tools: 计算长度偏置（初始长度-原长）
def l_calulator(theta_1, theta_2):
    a_1 = 0.25
    a_2 = 0.25
    b_1 = 0.21213
    b_2 = 0.1
    d_1 = 0.06
    d_2 = 0.10

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

    # 计算长度 l1 和 l2
    l_1 = np.sqrt((A_x_O - C_x_O)**2 + (A_y_O - C_y_O)**2)
    l_2 = np.sqrt((B_x_O - D_x_O)**2 + (B_y_O - D_y_O)**2)

    return l_1, l_2

# Tools: 计算驱动器设置初始偏置 (geom 模型)
def bias_calculator(l_10, l_20):
  l_1, l_2 = l_calulator(math.pi/2, math.pi/2)
  l1_rel = l_10 - l_1
  l2_rel = l_20 - l_2
  return l1_rel/2, l2_rel/2

# 优化后的参数
k1 = 385.1225
k2 = 335.4172
l10 = 0.1862
l20 = 0.2709
c1 = 24.1515
c2 = 10.3101
s1 = 0.00063079
s2 = 0.00060325
c1_thigh = 4.9746
c2_calf = 0.3958

m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)

RB_BAA_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_BAA_SlideJoint")
RB_BAA_FM_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_BAA_FM_SlideJoint")
RB_MAA_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_MAA_SlideJoint")
RB_MAA_FM_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_MAA_FM_SlideJoint")

RB_shoulder_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_shoulder")
RB_Elbow_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_Elbow")

# k
m.jnt_stiffness[RB_BAA_SlideJoint_id] = k2*2  # N·m/rad
m.jnt_stiffness[RB_BAA_FM_SlideJoint_id] = k2*2  # N·m/rad
m.jnt_stiffness[RB_MAA_SlideJoint_id] = k1*2  # N·m/rad
m.jnt_stiffness[RB_MAA_FM_SlideJoint_id] = k1*2  # N·m/rad

# c
m.dof_damping[RB_BAA_SlideJoint_id] = c2*2  # N·s/m
m.dof_damping[RB_BAA_FM_SlideJoint_id] = c2*2  # N·s/m
m.dof_damping[RB_MAA_SlideJoint_id] = c1*2  # N·s/m
m.dof_damping[RB_MAA_FM_SlideJoint_id] = c1*2  # N·s/m

# l0
MAA_bias, BAA_bias = bias_calculator(l10, l20)
m.qpos_spring[RB_BAA_SlideJoint_id] = BAA_bias  # rad
m.qpos_spring[RB_BAA_FM_SlideJoint_id] = BAA_bias  # rad
m.qpos_spring[RB_MAA_SlideJoint_id] = MAA_bias  # rad
m.qpos_spring[RB_MAA_FM_SlideJoint_id] = MAA_bias  # rad

# set the damping
m.dof_damping[RB_shoulder_id] = c1_thigh  # N·s/m
m.dof_damping[RB_Elbow_id] = c2_calf  # N·s/m

# Init the model
print(m.body_mass)
d.ctrl = np.zeros_like(d.ctrl)
mujoco.mj_step(m, d)  # update!

ExpResultList = []
viewer_flag = True

if viewer_flag:
  with mujoco.viewer.launch_passive(m, d) as viewer:
    input("Press to start!")
    start = time.time() if viewer_flag else d.time
    try:
      while (time.time() if viewer_flag else d.time) - start < 20: # viewer.is_running() and
        # print(time.time()-start if viewer_flag else d.time)
        step_time = time.time() if viewer_flag else d.time

        if step_time - start > 0 and step_time - start < 10: # 稳定时间
          d.ctrl[:2] = 40*1e3*s1
          d.ctrl[2:] = 00*1e3*s2

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