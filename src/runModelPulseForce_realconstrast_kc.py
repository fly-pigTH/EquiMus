# 气压点转移A2B sim vs real对比实验框架，用于计算参数c1, c2
# 批量生成数据，并直接进行优化
# 同时辨识 kc
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
import matplotlib.gridspec as gridspec


# 真实实验数据处理
start_time = time.time()
start_step_time = 5
pressure_group_num = 6
data_real_path = './data/Archive/A2Bdata/ToCenter_2025-02-20_21-06-20_A2B_data.csv'

# 原始数据：角度为 deg, 具有 pressure1_start, pressure2_start, pressure1_end, pressure2_end, relative_time, thigh_theta21, calf_theta32
data_real = pd.read_csv(data_real_path)    # Goup 3
data_real["experiment_group"] = data_real.groupby(["pressure1_start", "pressure2_start", "pressure1_end", 
                                     "pressure2_end"]).ngroup()
group_data_real_dic = {}    # using group_id to get the group_data_real
expdata_real_all = []
for group_id in data_real["experiment_group"].unique():
    # 访问某组数据
    group_data_real = data_real[data_real["experiment_group"]==group_id]
    # 访问时间
    group_data_real = group_data_real[group_data_real["relative_time"]>=start_step_time]
    pressure1_start = group_data_real["pressure1_start"].iloc[0]    # 访问group 参数
    pressure2_start = group_data_real["pressure2_start"].iloc[0]
    pressure1_end = group_data_real["pressure1_end"].iloc[0]
    pressure2_end = group_data_real["pressure2_end"].iloc[0]
    time_real = np.array(group_data_real["relative_time"]-start_step_time)  # 访问实验数据
    theta1_real = np.array(group_data_real["thigh_theta21"])
    theta2_real = np.array(group_data_real["calf_theta32"])

    t_intervel = np.linspace(0, 4, 300)
    theta1_real_intervel = np.interp(t_intervel, time_real, theta1_real)
    theta2_real_intervel = np.interp(t_intervel, time_real, theta2_real)
    expdata_real = np.stack((t_intervel, theta1_real_intervel, theta2_real_intervel), axis=0) 
    expdata_real_all.append(expdata_real)

expdata_real_all = np.array(expdata_real_all)
print(f"expdata_real_all.shape: {expdata_real_all.shape}")

# real experiment setting
data = pd.read_csv("./log/realdata/StaticProcess/real_StaticPoint_6group_2025-02-21_08-55-07.csv")
P1_array = data['P1'].values
P2_array = data['P2'].values
theta1_array = data['theta1'].values
theta2_array = data['theta2'].values
P1_array_all = np.linspace(0, 50, 6)
P2_array_all = np.linspace(0, 50, 6)

# 实验数据
# theta1_array = np.array(theta1_array, dtype=np.float32)*np.pi/180  # 输入 theta1
# theta2_array = np.array(theta2_array, dtype=np.float32)*np.pi/180  # 输入 theta2
P1_array = np.array(P1_array, dtype=np.float32)*1000  # 实验输出 P
P2_array = np.array(P2_array, dtype=np.float32)*1000  # 实验输出 P

effective_idx = np.array((P1_array/10000) * pressure_group_num + (P2_array/10000), dtype=np.int16)
print(effective_idx)

expdata_real_effective = expdata_real_all[effective_idx]
print(f"expdata_real_effective Shape = {expdata_real_effective.shape}")


# 仿真数据采集
path = "./models/v2_4/urdf/dog2_4singleLeg_realconstrast.xml"

ExpName = f"AtoB_realsim_constrast_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)
RB_BAA_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_BAA_SlideJoint")
RB_BAA_FM_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_BAA_FM_SlideJoint")
RB_MAA_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_MAA_SlideJoint")
RB_MAA_FM_SlideJoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_MAA_FM_SlideJoint")

# print(RB_BAA_SlideJoint_id, RB_BAA_FM_SlideJoint_id, RB_MAA_SlideJoint_id, RB_MAA_FM_SlideJoint_id)


# 遍历
# P1_array = np.linspace(0, 50, 6)
# P2_array = np.linspace(0, 50, 6)

P1, P2 = 30*10**3, 10*10**3
s1 = 0.0003538647203395965
s2 = 0.00034663180121508557

def run_single_experiment(params, P1, P2, save=False):
  """
    单次实验运行函数
    :param params: 包含(damping_MAA, damping_BAA, exp_id)的元组
  """
  exp_id, damping_MAA, damping_BAA, stiffness_MAA, stiffness_BAA, l10, l20 = params
  ForcePulse = np.array([P1*s1, P2*s2])
  
  # load the model
  m = mujoco.MjModel.from_xml_path(path)
  d = mujoco.MjData(m)

  # set the damping
  m.dof_damping[RB_BAA_SlideJoint_id] = damping_BAA  # N·s/m
  m.dof_damping[RB_BAA_FM_SlideJoint_id] = damping_BAA  # N·s/m
  m.dof_damping[RB_MAA_SlideJoint_id] = damping_MAA  # N·s/m
  m.dof_damping[RB_MAA_FM_SlideJoint_id] = damping_MAA  # N·s/m
  # k
  m.jnt_stiffness[RB_BAA_SlideJoint_id] = stiffness_BAA  # N·m/rad
  m.jnt_stiffness[RB_BAA_FM_SlideJoint_id] = stiffness_BAA  # N·m/rad
  m.jnt_stiffness[RB_MAA_SlideJoint_id] = stiffness_MAA  # N·m/rad
  m.jnt_stiffness[RB_MAA_FM_SlideJoint_id] = stiffness_MAA  # N·m/rad
  # l0
  m.qpos_spring[RB_BAA_SlideJoint_id] = l10  # rad
  m.qpos_spring[RB_BAA_FM_SlideJoint_id] = l10  # rad
  m.qpos_spring[RB_MAA_SlideJoint_id] = l20  # rad
  m.qpos_spring[RB_MAA_FM_SlideJoint_id] = l20  # rad



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

        # Trick: 由于实际数据稳定，所以这里只记录10s后的模拟数据，如果开始不收敛一定10s后的误差也很大
        Position = np.array(d.qpos)
        res = np.hstack((d.time, Position))
        ExpResultList.append(res)

      mujoco.mj_step(m, d)  # update!

  # # 按住ctrl C退出循环
  except KeyboardInterrupt:
    pass
  # print((res[1]+math.pi/2)*180/math.pi, end=" ")
  # print((res[2]+math.pi/2)*180/math.pi)
  result_array = np.array(ExpResultList)

  if save:
    # 创建文件夹
    if not os.path.exists(f"./data/SimConstrastResults/{ExpName}"):
      os.makedirs(f"./data/SimConstrastResults/{ExpName}")
    filename = f"./data/SimConstrastResults/{ExpName}/exp{exp_id:05d}.npz"
    # print(filename)
    np.savez(filename, result_array)

  # data process
  time_sim = result_array[:,0]-10
  theta1_sim = (result_array[:,1]+math.pi/2)*180/np.pi
  theta2_sim = (result_array[:,2]+math.pi/2)*180/np.pi
  return time_sim, theta1_sim, theta2_sim

# flag_idx = 0
global data_mode
data_mode = "effective"   # TODO: set here
def error(params, mode="train", data_mode=data_mode):
  # time_start = time.time()
  # global flag_idx
  # flag_idx += 1
  # print(flag_idx)
  c1, c2, k1, k2, l10, l20 = params
  expdata_sim_all = []
  if data_mode == "effective":
    for p1, p2 in zip(P1_array, P2_array):
      time_sim, theta1_sim, theta2_sim = run_single_experiment((0, c1, c2, k1, k2, l10, l20), p1, p2, save=False)
      t_intervel = np.linspace(0, 4, 300)
      theta1_sim_intervel = np.interp(t_intervel, time_sim, theta1_sim)
      theta2_sim_intervel = np.interp(t_intervel, time_sim, theta2_sim)
      expdata_sim = np.stack((t_intervel, theta1_sim_intervel, theta2_sim_intervel), axis=0) 
      expdata_sim_all.append(expdata_sim)
    expdata_sim_all = np.array(expdata_sim_all)
    errorMat = expdata_sim_all - expdata_real_effective
    SE_mat = errorMat**2
    MSE_mat = np.mean(SE_mat, axis=2)
    # theta1 和 theta2 的总体MSE
    MSE_mat_overview = np.mean(MSE_mat, axis=0)
    MSE_all = np.mean(MSE_mat_overview[1:])
  elif data_mode == "all":
    for p1 in P1_array_all:
      for p2 in P2_array_all:
        time_sim, theta1_sim, theta2_sim = run_single_experiment((0, c1, c2), p1, p2, save=False)
        t_intervel = np.linspace(0, 4, 300)
        theta1_sim_intervel = np.interp(t_intervel, time_sim, theta1_sim)
        theta2_sim_intervel = np.interp(t_intervel, time_sim, theta2_sim)
        expdata_sim = np.stack((t_intervel, theta1_sim_intervel, theta2_sim_intervel), axis=0) 
        expdata_sim_all.append(expdata_sim)
    expdata_sim_all = np.array(expdata_sim_all)
    errorMat = expdata_sim_all - expdata_real_all
    SE_mat = errorMat**2
    MSE_mat = np.mean(SE_mat, axis=2)
    # theta1 和 theta2 的总体MSE
    MSE_mat_overview = np.mean(MSE_mat, axis=0)
    MSE_all = np.mean(MSE_mat_overview[1:])
  if mode == "train":
    return MSE_all
  else:
    return MSE_mat_overview


import matplotlib
matplotlib.use("TkAgg")  # 指定 TkAgg 后端，避免 macOS 线程问题
# 设置全局字体属性
plt.rcParams['font.family'] = 'Arial'  # 设置字体为 Arial
plt.rcParams['font.size'] = 15  # 设置全局字体大小

# draw the optimal solution
if __name__ != "__main__":
  para = (0, 291.5757670770301, 98.3306672255234)
  expdata_sim_all = []
  for p1, p2 in zip(P1_array, P2_array):
  # for p1 in P1_array:
  #   for p2 in P2_array:
      time_sim, theta1_sim, theta2_sim = run_single_experiment(para, p1, p2, save=False)
      t_intervel = np.linspace(0, 4, 300)
      theta1_sim_intervel = np.interp(t_intervel, time_sim, theta1_sim)
      theta2_sim_intervel = np.interp(t_intervel, time_sim, theta2_sim)
      expdata_sim = np.stack((t_intervel, theta1_sim_intervel, theta2_sim_intervel), axis=0) 
      expdata_sim_all.append(expdata_sim)
  expdata_sim_all = np.array(expdata_sim_all)
  exp_id = 0

  # draw the comparison
  # 假设expdata_real_effective和expdata_sim_all是已经定义好的数据
  num_experiments = len(expdata_real_effective)
  ncols = 5  # 每行5张图
  nrows = math.ceil(num_experiments / ncols)  # 根据实验数计算行数

  # 创建一个统一的网格布局
  fig = plt.figure(figsize=(20, 10))  # 设置整个图形的大小
  gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.05, hspace=0.1, left=0.08, right=0.99, top=0.9, bottom=0.1)  # 设置子图间距

  # 绘制每个实验的图形
  for exp_id in range(num_experiments):
      ax = fig.add_subplot(gs[exp_id // ncols, exp_id % ncols])
      expdata_real, expdata_sim = expdata_real_effective[exp_id], expdata_sim_all[exp_id]
      if exp_id == 0:
        ax.plot(expdata_real[0], expdata_real[1], label=r"$\theta_1-Real$", color='blue', linestyle='-', alpha=0.6)
        ax.plot(expdata_real[0], expdata_real[2], label=r"$\theta_2-Real$", color='orange', linestyle='-', alpha=0.6)
        ax.plot(expdata_sim[0], expdata_sim[1], label=r"$\theta_1-Sim$", color='green', linestyle='-', alpha=0.6)
        ax.plot(expdata_sim[0], expdata_sim[2], label=r"$\theta_2-Sim$", color='red', linestyle='-', alpha=0.6)
      else:
        ax.plot(expdata_real[0], expdata_real[1], color='blue', linestyle='-', alpha=0.6)
        ax.plot(expdata_real[0], expdata_real[2], color='orange', linestyle='-', alpha=0.6)
        ax.plot(expdata_sim[0], expdata_sim[1], color='green', linestyle='-', alpha=0.6)
        ax.plot(expdata_sim[0], expdata_sim[2], color='red', linestyle='-', alpha=0.6)
      # ax.set_title(f"Experiment {exp_id + 1}")
      # 在右上角添加标注
      ax.text(0.95, 0.95, f'[{int(P1_array[exp_id]/1000):d}, {int(P2_array[exp_id]/1000):d}]->[0, 0] kPa', transform=ax.transAxes, 
              ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
      # text: RMSE
      RMSE_THETA1 = np.sqrt(np.mean((expdata_real[1]-expdata_sim[1])**2))
      RMSE_THETA2 = np.sqrt(np.mean((expdata_real[2]-expdata_sim[2])**2))
      ax.text(0.95, 0.75, f"RMSE1: {RMSE_THETA1:.4f}", transform=ax.transAxes, 
              ha='right', va='top', fontsize=12)
      ax.text(0.95, 0.45, f"RMSE2: {RMSE_THETA2:.4f}", transform=ax.transAxes,
              ha='right', va='top', fontsize=12)
      
      ax.grid(alpha=0.4)  # 添加网格线
      ax.set_xlim(0, 4)  # 设置x轴范围
      ax.set_ylim(0, 120)  # 设置y轴范围
      ax.set_xticks(np.arange(0, 4.1, 1))  # 设置x轴刻度
      ax.set_yticks(np.arange(0, 120.1, 30))  # 设置y轴刻度
      if exp_id%ncols != 0:   # 如果不在最左边
        ax.set_yticklabels([])  # 隐藏y轴刻度
      if exp_id//ncols != nrows-1:  # 如果不在最下面
        ax.set_xticklabels([])  # 隐藏x轴刻度


  # 设置统一的x轴和y轴标签
  fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=15)
  fig.text(0.04, 0.5, 'Angle (deg)', va='center', rotation='vertical', fontsize=15)

  # 添加全图图名
  fig.suptitle('Dynamic Test Comparison: Real vs Simulated Data', fontsize=25, y=0.95)

  # 添加统一的图例
  handles, labels = [], []
  for ax in fig.axes:
    for handle, label in zip(*ax.get_legend_handles_labels()):
      handles.append(handle)
      labels.append(label)
  fig.legend(handles, labels, bbox_to_anchor=(0.95, 0.3), ncol=1, fontsize=15)  # 添加图例

  # 调整子图间距
  plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整子图间距，留出空间给图例
  plt.show()


# train or test
if __name__ == "__main__":

  exp_mode = "train"  # train or test
  # data_mode = "all"  # all or effictive

  if exp_mode == "train":
    global time_start, time_epi_start, epi
    time_start = time.time()
    time_epi_start = time.time()
    epi = 0

    # 使用进化算法进行参数优化
    # 记录进度
    def callback(xk, convergence):
      global time_epi_start, time_epi_start, epi
      epi += 1
      time_cost_epi = time.time() - time_epi_start
      time_epi_start = time.time()
      print(f"Epoch{epi}: Optimal c1={xk[0]:.4f}, c2={xk[1]:.4f}, k1={xk[2]:.4f}, k2={xk[3]:.4f}, l10={xk[4]:.4f}, l20={xk[5]:.4f}, MSE={error(xk):.4f}, Convergence={convergence:.6f}", end=" | ")
      print(f"Time/epoch: {time_cost_epi:.2f}s")
      

    from scipy.optimize import differential_evolution
    bounds = [(0, 300), (0, 300), (0, 1000), (0, 1000), (-1, 1), (-1, 1)]  # c1, c2, k1, k2, l10, l20 的范围

    result = differential_evolution(error, bounds, strategy='best1bin', maxiter=1000, disp=False, callback=callback, popsize=10)

    best_c1, best_c2, best_k1, best_k2, best_l10, best_l20 = result.x
    print(f"Optimal c1={best_c1:.4f}, c2={best_c2:.4f}, k1={best_k1:.4f}, k2={best_k2:.4f}, l10={best_l10:.4f}, l20={best_l20:.4f}, MSE={result.fun:.4f}")
    time_cost = time.time() - time_start
    print(f"Time Cost: {time_cost:.2f}s")
  
  elif exp_mode == "test":
    # 使用最优参数进行测试
    best_c1, best_c2 = 291.5757670770301, 98.3306672255234
    MSE_mat_overview = error([best_c1, best_c2], mode=exp_mode)
    print(MSE_mat_overview)