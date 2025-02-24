# 模拟10N到自然的过程，和实物对比
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
import itertools
from tqdm import tqdm

start_time = time.time()

# path = "./models/SingleLeg_ideal.xml"
path = "./models/v2_4/urdf/dog2_4singleLeg_realconstrast.xml"

ExpName = f"AtoB_realsim_constrast_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)
RB_BAA_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_BAA_SlideJoint")
RB_BAA_FM_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_BAA_FM_SlideJoint")
RB_MAA_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_MAA_SlideJoint")
RB_MAA_FM_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_MAA_FM_SlideJoint")

# print(RB_BAA_SlideJoint_id, RB_BAA_FM_SlideJoint_id, RB_MAA_SlideJoint_id, RB_MAA_FM_SlideJoint_id)

# input()
P1, P2 = 30*10**3, 10*10**3
s1 = 0.0003538647203395965
s2 = 0.00034663180121508557

P1_array = np.array([P1])
P2_array = np.array([P2])


def run_single_experiment(params, P1, P2, save=True):
  """
    单次实验运行函数
    :param params: 包含(damping_MAA, damping_BAA, exp_id)的元组
  """
  exp_id, damping_MAA, damping_BAA = params
  ForcePulse = np.array([P1*s1, P2*s2])
  
  # load the model
  m = mujoco.MjModel.from_xml_path(path)
  d = mujoco.MjData(m)

  # set the damping
  m.dof_damping[RB_BAA_SlideJoint_id] = damping_BAA  # N·s/m
  m.dof_damping[RB_BAA_FM_SlideJoint_id] = damping_BAA  # N·s/m
  m.dof_damping[RB_MAA_SlideJoint_id] = damping_MAA  # N·s/m
  m.dof_damping[RB_MAA_FM_SlideJoint_id] = damping_MAA  # N·s/m

  # Init the model
  # print(m.body_mass)
  d.ctrl = np.zeros_like(d.ctrl)
  mujoco.mj_step(m, d)  # update!

  # Experiment Settings
  ExpResultList = []
  viewer_flag = 0

  start = time.time() if viewer_flag else d.time
  try:
    while (time.time() if viewer_flag else d.time) - start < 20: # viewer.is_running() and
      # print(time.time()-start if viewer_flag else d.time)
      step_time = time.time() if viewer_flag else d.time

      if step_time - start > 0 and step_time - start < 10: # 稳定时间
        d.ctrl[:2] = ForcePulse[0]
        d.ctrl[2:] = ForcePulse[1]

      if 10 <= step_time - start < 20: # 稳定时间
        d.ctrl[:2] = 0
        d.ctrl[2:] = 0

      Position = np.array(d.qpos)
      res = np.hstack(([damping_MAA, damping_BAA, P1, P2, d.time], Position))
      ExpResultList.append(res)
      mujoco.mj_step(m, d)  # update!

  # # 按住ctrl C退出循环
  except KeyboardInterrupt:
    pass
  print((res[5]+math.pi/2)*180/math.pi, end=" ")
  print((res[6]+math.pi/2)*180/math.pi)
  result_array = np.array(ExpResultList)
  # 创建文件夹
  if not os.path.exists(f"./data/SimConstrastResults/{ExpName}"):
    os.makedirs(f"./data/SimConstrastResults/{ExpName}")
  filename = f"./data/SimConstrastResults/{ExpName}/exp{exp_id:05d}.npz"
  # print(filename)
  np.savez(filename, result_array)

  return (exp_id, filename, damping_MAA, damping_BAA)


if __name__ == "__main__":
  # 参数范围定义
  damping_values = np.linspace(1, 100, 20)  # 100个阻尼值

  # 生成全排列组合
  damping_combinations = list(itertools.product(damping_values, repeat=2))
  # print(damping_combinations)

  # 添加实验ID
  paras = [(i, maa, baa) for i, (maa, baa) in enumerate(damping_combinations)]
  print(paras)
  
  for para in tqdm(paras):
    # print(para[0])
    for p1 in P1_array:
      for p2 in P2_array:
        run_single_experiment(para, p1, p2, save=True)
    # time.sleep(0.1)
  end_time = time.time()
  print("Time Cost: ", end_time-start_time)