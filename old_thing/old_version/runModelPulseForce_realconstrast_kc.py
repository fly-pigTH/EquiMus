# 参数辨识测试框架（整合版本）
# 气压点转移A2B sim vs real对比实验框架，用于计算参数c1, c2
# 批量生成数据，并直接进行优化
# 同时辨识 kc，关节的c也考虑
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
from matplotlib.widgets import Slider
import scipy

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

    t_intervel = np.linspace(0, 2, 201)
    # theta1_real_intervel = np.interp(t_intervel, time_real, theta1_real)
    # theta2_real_intervel = np.interp(t_intervel, time_real, theta2_real)
    theta1_spline = scipy.interpolate.make_interp_spline(time_real, theta1_real, k=3)
    theta2_spline = scipy.interpolate.make_interp_spline(time_real, theta2_real, k=3)
    theta1_real_intervel = theta1_spline(t_intervel)
    theta2_real_intervel = theta2_spline(t_intervel) 
    expdata_real = np.stack((t_intervel, theta1_real_intervel, theta2_real_intervel), axis=0)
    expdata_real_all.append(expdata_real)

expdata_real_all = np.array(expdata_real_all)
print(f"expdata_real_all.shape: {expdata_real_all.shape}")

# real experiment setting
# data = pd.read_csv("./log/realdata/StaticProcess/real_StaticPoint_6group_2025-02-21_08-55-07.csv")
# P1_array = data['P1'].values
# P2_array = data['P2'].values
# theta1_array = data['theta1'].values
# theta2_array = data['theta2'].values

data = pd.read_csv("src/physical_verify/static/data/real_static_state/real(all)_StaticPoint_6group_2025-02-21_08-55-07_2025-08-01_19-34-21.csv")
P1_array = np.array(data['P1 (kPa)'].values, dtype=np.float32)
P2_array = np.array(data['P2 (kPa)'].values, dtype=np.float32)
theta1_array = np.array(data['theta1 (deg)'].values, dtype=np.float32)
theta2_array = np.array(data['theta2 (deg)'].values, dtype=np.float32)

P1_array_all = np.linspace(0, 50, 6)
P2_array_all = np.linspace(0, 50, 6)

# 实验数据
# theta1_array = np.array(theta1_array, dtype=np.float32)*np.pi/180  # 输入 theta1
# theta2_array = np.array(theta2_array, dtype=np.float32)*np.pi/180  # 输入 theta2
P1_array = np.array(P1_array, dtype=np.float32)*1000  # 实验输出 P
P2_array = np.array(P2_array, dtype=np.float32)*1000  # 实验输出 P

effective_idx = np.array((P1_array/10000) * pressure_group_num + (P2_array/10000), dtype=np.int16)
print(effective_idx)

# effective_idx = [19]  # TODO 30,10

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

RB_shoulder_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_shoulder")
RB_Elbow_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "RB_Elbow")

# print(RB_BAA_SlideJoint_id, RB_BAA_FM_SlideJoint_id, RB_MAA_SlideJoint_id, RB_MAA_FM_SlideJoint_id)


# 遍历
# P1_array = np.linspace(0, 50, 6)
# P2_array = np.linspace(0, 50, 6)

P1, P2 = 30*10**3, 10*10**3
# s1 = 0.0003538647203395965
# s2 = 0.00034663180121508557

# TODO: consider the l0 calculator
def run_single_experiment(params, P1, P2, save=False):
  """
    单次实验运行函数
    :param params: 包含(damping_MAA, damping_BAA, exp_id)的元组
  """
  try:
    stiffness_MAA, stiffness_BAA, l10, l20, damping_MAA, damping_BAA, s1, s2, c1_thigh, c2_calf = params

    ForcePulse = np.array([P1*s1, P2*s2])
    # print(f"ForcePulse: {ForcePulse}")
    
    # load the model
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)

    # set the damping
    # k
    m.jnt_stiffness[RB_BAA_SlideJoint_id] = stiffness_BAA*2  # N·m/rad
    m.jnt_stiffness[RB_BAA_FM_SlideJoint_id] = stiffness_BAA*2  # N·m/rad
    m.jnt_stiffness[RB_MAA_SlideJoint_id] = stiffness_MAA*2  # N·m/rad
    m.jnt_stiffness[RB_MAA_FM_SlideJoint_id] = stiffness_MAA*2  # N·m/rad

    # c
    m.dof_damping[RB_BAA_SlideJoint_id] = damping_BAA*2  # N·s/m
    m.dof_damping[RB_BAA_FM_SlideJoint_id] = damping_BAA*2  # N·s/m
    m.dof_damping[RB_MAA_SlideJoint_id] = damping_MAA*2  # N·s/m
    m.dof_damping[RB_MAA_FM_SlideJoint_id] = damping_MAA*2  # N·s/m
    
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
    d.ctrl = np.zeros_like(d.ctrl)
    mujoco.mj_step(m, d)  # update!

    # Experiment Settings
    ExpResultList = []
    viewer_flag = 0
    time_exp = 20
    time_step = 10

    start = time.time() if viewer_flag else d.time
    try:
      while (time.time() if viewer_flag else d.time) - start < time_exp: # viewer.is_running() and
        # print(time.time()-start if viewer_flag else d.time)
        step_time = time.time() if viewer_flag else d.time

        if step_time - start > 0 and step_time - start < 10: # 稳定时间
          d.ctrl[:2] = ForcePulse[0]
          d.ctrl[2:] = ForcePulse[1]

        if time_step <= step_time - start < time_exp: # 稳定时间
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
    time_sim = result_array[:,0]-time_step
    theta1_sim = (result_array[:,1]+math.pi/2)*180/np.pi
    theta2_sim = (result_array[:,2]+math.pi/2)*180/np.pi
    return time_sim, theta1_sim, theta2_sim
  except Exception as e:
    print(f"Error: {e}")
    return np.zeros(400), np.zeros(400), np.zeros(400)

# flag_idx = 0
global data_mode
data_mode = "effective"   # TODO: set here

def smooth_velocity(velocity, window_size=5):
    """
    对速度数据进行平滑处理
    Args:
        velocity: 速度数据，shape为(n_experiments, n_joints, n_timesteps)
        window_size: 滑动窗口大小
    Returns:
        smoothed_velocity: 平滑后的速度数据
    """
    return np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='same'),
        axis=2,
        arr=velocity
    )

def error(params, mode="train", data_mode=data_mode):
  # print("R", end='')
  # time_start = time.time()
  # global flag_idx
  # flag_idx += 1
  # print(flag_idx)
  k1, k2, l10, l20, c1, c2, s1, s2, c1_thigh, c2_calf = params
  expdata_sim_all = []
  if data_mode == "effective":
    for p1, p2 in zip(P1_array[:], P2_array[:]):
      time_sim, theta1_sim, theta2_sim = run_single_experiment((k1, k2, l10, l20, c1, c2, s1, s2, c1_thigh, c2_calf), p1, p2, save=False)
      t_intervel = np.linspace(0, 2, 201)
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

    # 增加 MSE of velocity
    # velocity
    # print(f"time_sim[1]-time_sim[0]: {time_sim[1]-time_sim[0]}")
    # not the time_sim[1]-time_sim[0]
    delta_t = t_intervel[1]-t_intervel[0]
    velocity_real = np.diff(expdata_real_effective, axis=2)/(delta_t)
    velocity_sim = np.diff(expdata_sim_all, axis=2)/(delta_t)
    # 对速度进行平滑
    velocity_real = smooth_velocity(velocity_real, window_size=5)
    velocity_sim = smooth_velocity(velocity_sim, window_size=5)
    # errorMat_velocity = velocity_sim - velocity_real
    # SE_mat_velocity = errorMat_velocity**2
    # MSE_mat_velocity = np.mean(SE_mat_velocity, axis=2)
    # 判断速度符号是否一致
    velocity_sim_sign = np.where(velocity_sim > 0, 1, -1)
    velocity_real_sign = np.where(velocity_real > 0, 1, -1)
    errorMat_velocity = np.where(velocity_sim_sign == velocity_real_sign, 0, 1)
    SE_mat_velocity = errorMat_velocity**2
    MSE_mat_velocity = np.mean(SE_mat_velocity, axis=2)
    MSE_mat_overview_velocity = np.mean(MSE_mat_velocity, axis=0)
    MSE_all_velocity = np.mean(MSE_mat_overview_velocity[1:])   # 只考虑theta1的速度
    print(f"MSE_all: {MSE_all:.4f}, MSE_all_velocity: {MSE_all_velocity:.4f}")

  # elif data_mode == "all":
  #   for p1 in P1_array_all:
  #     for p2 in P2_array_all:
  #       time_sim, theta1_sim, theta2_sim = run_single_experiment((0, c1, c2), p1, p2, save=False)
  #       t_intervel = np.linspace(0, 2, 201)
  #       theta1_sim_intervel = np.interp(t_intervel, time_sim, theta1_sim)
  #       theta2_sim_intervel = np.interp(t_intervel, time_sim, theta2_sim)
  #       expdata_sim = np.stack((t_intervel, theta1_sim_intervel, theta2_sim_intervel), axis=0) 
  #       expdata_sim_all.append(expdata_sim)
  #   expdata_sim_all = np.array(expdata_sim_all)
  #   errorMat = expdata_sim_all - expdata_real_all
  #   SE_mat = errorMat**2
  #   MSE_mat = np.mean(SE_mat, axis=2)
  #   # theta1 和 theta2 的总体MSE
  #   MSE_mat_overview = np.mean(MSE_mat, axis=0)
  #   MSE_all = np.mean(MSE_mat_overview[1:])
  if mode == "train":
    return MSE_all + 500*MSE_all_velocity
  else:
    print(f"MSE_mat_overview: {MSE_mat_overview}")
    print(f"MSE_mat_overview_velocity: {MSE_mat_overview_velocity}")
    return MSE_mat_overview


import matplotlib
matplotlib.use("TkAgg")  # 指定 TkAgg 后端，避免 macOS 线程问题
# 设置全局字体属性
plt.rcParams['font.family'] = 'Arial'  # 设置字体为 Arial
plt.rcParams['font.size'] = 15  # 设置全局字体大小


# 测试接口是否有效
if __name__ != "__main__":
  # stiffness_MAA, stiffness_BAA, damping_MAA, damping_BAA, s1, s2
  k1=130.4233
  k2=117.9855
  l10=0.1640
  l20=0.2580
  c1=99.6650
  c2=42.6441
  s1=0.00039486
  s2=0.00044950
  c1_thigh = 0.0
  c2_calf = 0.0

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

  para = [k1, k2, l10, l20, c1, c2, s1, s2, c1_thigh, c2_calf]
  time_sim, theta1_sim, theta2_sim = run_single_experiment(para, 30*10**3, 10*10**3, save=False)

  # 创建图形和轴
  fig, ax = plt.subplots()
  plt.subplots_adjust(left=0.1, bottom=0.6)  # 调整布局，为滑动杆留出空间  


  l_sim1, = plt.plot(time_sim, theta1_sim, label="theta1_sim")
  l_sim2, = plt.plot(time_sim, theta2_sim, label="theta2_sim")
  # realdata
  exp_id = 19  # 19
  plt.plot(expdata_real_all[exp_id][0], expdata_real_all[exp_id][1], label="theta1_real")
  plt.plot(expdata_real_all[exp_id][0], expdata_real_all[exp_id][2], label="theta2_real")
  plt.xlabel("Time (s)")
  plt.ylabel("Angle (deg)")
  plt.legend()
  plt.xlim(0, 4)

  print()

  # 创建滑动杆轴
  # 滑动杆位置定义
  ax_k1 = plt.axes([0.25, 0.45, 0.65, 0.03])  # k1 滑动杆位置
  ax_k2 = plt.axes([0.25, 0.4, 0.65, 0.03])  # k2 滑动杆位置
  ax_l1 = plt.axes([0.25, 0.35, 0.65, 0.03])  # l1 滑动杆位置
  ax_l2 = plt.axes([0.25, 0.30, 0.65, 0.03])  # l2 滑动杆位置
  ax_s1 = plt.axes([0.25, 0.25, 0.65, 0.03])  # s1 滑动杆位置
  ax_s2 = plt.axes([0.25, 0.20, 0.65, 0.03])  # s2 滑动杆位置
  ax_c1 = plt.axes([0.25, 0.15, 0.65, 0.03])  # c1 滑动杆位置
  ax_c2 = plt.axes([0.25, 0.1, 0.65, 0.03])  # c2 滑动杆位置
  ax_c11 = plt.axes([0.25, 0.05, 0.65, 0.03])  # c2 滑动杆位置
  ax_c12 = plt.axes([0.25, 0.00, 0.65, 0.03])  # c2 滑动杆位置


  # 创建滑动杆
  k1_slider = Slider(ax_k1, 'k1', 0, 700, valinit=k1)
  k2_slider = Slider(ax_k2, 'k2', 0, 700, valinit=k2)
  l1_slider = Slider(ax_l1, 'l1', 0, 1, valinit=l10)
  l2_slider = Slider(ax_l2, 'l2', 0, 1, valinit=l20)
  s1_slider = Slider(ax_s1, 's1', 0, 0.005, valinit=s1)
  s2_slider = Slider(ax_s2, 's2', 0, 0.005, valinit=s2)
  c1_slider = Slider(ax_c1, 'c1', 0, 100, valinit=c1)
  c2_slider = Slider(ax_c2, 'c2', 0, 100, valinit=c2)
  c1_thigh_slider = Slider(ax_c11, 'c1_thigh', 0, 100, valinit=c1_thigh)
  c2_calf_slider = Slider(ax_c12, 'c2_calf', 0, 10, valinit=c2_calf)

  # 更新函数
  def update(val):
    # 获取滑动杆的值
    c1 = c1_slider.val
    c2 = c2_slider.val

    # 更新参数列表
    para[0] = k1_slider.val
    para[1] = k2_slider.val
    para[2] = l1_slider.val
    para[3] = l2_slider.val
    para[6] = s1_slider.val
    para[7] = s2_slider.val
    para[4] = c1
    para[5] = c2
    para[8] = c1_thigh_slider.val
    para[9] = c2_calf_slider.val


    # 重新运行仿真
    time_sim, theta1_sim, theta2_sim = run_single_experiment(para, 30 * 10**3, 10 * 10**3, save=False)

    # 更新曲线数据
    l_sim1.set_ydata(theta1_sim)
    l_sim2.set_ydata(theta2_sim)

    # 重绘图
    fig.canvas.draw_idle()

  # 绑定滑动杆事件
  k1_slider.on_changed(update)
  k2_slider.on_changed(update)
  l1_slider.on_changed(update)
  l2_slider.on_changed(update)
  s1_slider.on_changed(update)
  s2_slider.on_changed(update)
  c1_slider.on_changed(update)
  c2_slider.on_changed(update)
  c1_thigh_slider.on_changed(update)
  c2_calf_slider.on_changed(update)

  plt.show()


# draw the optimal solution
if __name__ != "__main__":
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

  k1 = 524.9779
  k2 = 281.8427
  l10 = 0.1893
  l20 = 0.2682
  c1 = 28.0952
  c2 = 27.5535
  s1 = 0.00084641
  s2 = 0.00081535
  c1_thigh = 2.8155
  c2_calf = 0.3083

  para = [k1, k2, l10, l20, c1, c2, s1, s2, c1_thigh, c2_calf]
  expdata_sim_all = []
  for p1, p2 in zip(P1_array, P2_array):
      time_sim, theta1_sim, theta2_sim = run_single_experiment(para, p1, p2, save=False)
      t_intervel = np.linspace(0, 2, 201)
      theta1_sim_intervel = np.interp(t_intervel, time_sim, theta1_sim)
      theta2_sim_intervel = np.interp(t_intervel, time_sim, theta2_sim)
      expdata_sim = np.stack((t_intervel, theta1_sim_intervel, theta2_sim_intervel), axis=0) 
      expdata_sim_all.append(expdata_sim)
  expdata_sim_all = np.array(expdata_sim_all)
  exp_id = 0

  print("Error: ", error(para, mode="test"))

  # draw the comparison
  # 假设expdata_real_effective和expdata_sim_all是已经定义好的数据
  num_experiments = len(expdata_real_effective)
  ncols = 5  # 每行5张图
  nrows = math.ceil(num_experiments / ncols)  # 根据实验数计算行数

  # 创建一个统一的网格布局
  fig = plt.figure(figsize=(25, 20))  # 设置整个图形的大小
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
if __name__== "__main__":

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
      print(f"Epoch: {epi}, Time/epoch: {time_cost_epi:.2f}s, convergence={convergence:.4f}, MSE={error(xk):.4f}")
      print(f"k1={xk[0]:.4f}, k2={xk[1]:.4f}, l10={xk[2]:.4f}, l20={xk[3]:.4f}, c1={xk[4]:.4f}, c2={xk[5]:.4f}, s1={xk[6]:.8f}, s2={xk[7]:.8f}, c1_thigh={xk[8]:.4f}, c2_calf={xk[9]:.4f}")
      print(f"Time/epoch: {time_cost_epi:.2f}s")
      # 运行当前参数的仿真
      time_sim, theta1_sim, theta2_sim = run_single_experiment(xk, 30*10**3, 10*10**3, save=False)
      t_intervel = np.linspace(0, 2, 201)
      theta1_sim_intervel = np.interp(t_intervel, time_sim, theta1_sim)
      theta2_sim_intervel = np.interp(t_intervel, time_sim, theta2_sim)
      
      # # 创建新图或清除现有图
      # plt.clf()
      
      # # 创建子图
      # fig = plt.gcf()
      # fig.set_size_inches(15, 6)
      
      # # 角度对比图
      # plt.subplot(121)
      # plt.plot(t_intervel, theta1_sim_intervel, 'g-', label=r'$\theta_1$ Sim', alpha=0.7)
      # plt.plot(t_intervel, theta2_sim_intervel, 'r-', label=r'$\theta_2$ Sim', alpha=0.7)
      # plt.plot(expdata_real_effective[0][0], expdata_real_effective[0][1], 'b--', label=r'$\theta_1$ Real', alpha=0.7)
      # plt.plot(expdata_real_effective[0][0], expdata_real_effective[0][2], 'y--', label=r'$\theta_2$ Real', alpha=0.7)
      # plt.xlabel('Time (s)')
      # plt.ylabel('Angle (deg)')
      # plt.title(f'Angle Comparison (Epoch {epi})')
      # plt.grid(True, alpha=0.3)
      # plt.legend()
      
      # # 速度对比图
      # plt.subplot(122)
      # dt = t_intervel[1] - t_intervel[0]
      # vel1_sim = np.diff(theta1_sim_intervel)/dt
      # vel2_sim = np.diff(theta2_sim_intervel)/dt
      # vel1_real = np.diff(expdata_real_effective[0][1])/dt
      # vel2_real = np.diff(expdata_real_effective[0][2])/dt
      # # 平滑
      # vel1_sim = np.convolve(vel1_sim, np.ones(5)/5, mode='same')
      # vel2_sim = np.convolve(vel2_sim, np.ones(5)/5, mode='same')
      # vel1_real = np.convolve(vel1_real, np.ones(5)/5, mode='same')
      # vel2_real = np.convolve(vel2_real, np.ones(5)/5, mode='same')
      # t_vel = t_intervel[:-1]
      
      # plt.plot(t_vel, vel1_sim, 'g-', label=r'$\dot{\theta}_1$ Sim', alpha=0.7)
      # plt.plot(t_vel, vel2_sim, 'r-', label=r'$\dot{\theta}_2$ Sim', alpha=0.7)
      # plt.plot(t_vel, vel1_real, 'b--', label=r'$\dot{\theta}_1$ Real', alpha=0.7)
      # plt.plot(t_vel, vel2_real, 'y--', label=r'$\dot{\theta}_2$ Real', alpha=0.7)
      # plt.xlabel('Time (s)')
      # plt.ylabel('Angular Velocity (deg/s)')
      # plt.title(f'Velocity Comparison (Epoch {epi})')
      # plt.grid(True, alpha=0.3)
      # plt.legend()
      
      # # 调整布局
      # plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # 为底部文本留出空间
      
      # # 在底部添加优化信息
      # info_text = f'MSE: {error(xk):.4f}\n'
      # info_text += f'k1={xk[0]:.1f}, k2={xk[1]:.1f}, l10={xk[2]:.4f}, l20={xk[3]:.4f}\n'
      # info_text += f'c1={xk[4]:.2f}, c2={xk[5]:.2f}, s1={xk[6]:.6f}, s2={xk[7]:.6f}, c1_thigh={xk[8]:.2f}, c2_calf={xk[9]:.2f}'
      # plt.figtext(0.1, 0.01, info_text, fontsize=10, ha='left', va='bottom')
      
      # plt.pause(0.01)  # 暂停一小段时间以显示图形
    from scipy.optimize import differential_evolution
    # 假设kls固定
    # s1 = 0.0003490837132109409
    # s2 = 0.0003815501584986775
    # k1 = 370
    # k2 = 302
    # l10 = 0.1641527577313371
    # l20 = 0.2579251802483194
    # c1_thigh = 0
    # c2_calf = 0

    # k1, k2 = 136.5468, 113.6052
    # l10, l20 = 0.1642, 0.2579
    # c1, c2 = 132.8775, 84.2973
    # s1, s2 = 0.00034908, 0.00038155
    # c1_thigh, c2_calf = 4.0915, 0.0033

    # ===== MAA 系列参数 =====
    m1 = 0.18648                # 质量 [kg]（原数据单位：g）
    l10 = 0.17437               # 原长 [m]（原数据单位：mm）
    c1 = 10.82                  # 阻尼系数 [Ns/m]
    k1 = 367.79         # 动态刚度 [N/m]
    k1_static = 373.59          # 静态刚度 [N/m]
    s1 = 6.54e-4                # 等效面积 [m²]（原数据单位：cm²，1 cm² = 1e-4 m²）

    # ===== BAA 系列参数 =====
    m2 = 0.27266                # 质量 [kg]
    l20 = 0.25357               # 原长 [m]
    c2 = 11.27                  # 阻尼系数 [Ns/m]
    k2 = 291.76         # 动态刚度 [N/m]
    k2_static = 302.89          # 静态刚度 [N/m]
    s2 = 6.37e-4                # 等效面积 [m²]

    # ===== 其他参数补充（根据示例需求）=====
    c1_thigh = 3.0              # 大腿附加阻尼（示例默认值）
    c2_calf = 1.0               # 小腿附加阻尼（示例默认值）
    bounds = [(k1*0.9, k1*1.5), (k2*0.9, k2*1.5), (l10*0.9, l10*1.1), (l20*0.9, l20*1.1), (0, 30), (0, 30), (s1*0.5, s1*1.5), (s2*0.5, s2*1.5), (0, 5), (0, 5)]  # 每个参数都有10%的变化范围
    result = differential_evolution(error, bounds, strategy='best1bin', maxiter=10000, disp=False, callback=callback, popsize=25)
    # print the res
    k1_opt, k2_opt, l10_opt, l20_opt, c1_opt, c2_opt, s1_opt, s2_opt, c1_thigh_opt, c2_calf_opt = result.x
    print(f"Optimal k1={k1_opt:.4f}, k2={k2_opt:.4f}, l10={l10_opt:.4f}, l20={l20_opt:.4f}, c1={c1_opt:.4f}, c2={c2_opt:.4f}, s1={s1_opt:.8f}, s2={s2_opt:.8f}, c1_thigh={c1_thigh_opt:.4f}, c2_calf={c2_calf_opt:.4f}, MSE={result.fun:.4f}")
    print(f"Error: {error([k1_opt, k2_opt, l10_opt, l20_opt, c1_opt, c2_opt, s1_opt, s2_opt, c1_thigh_opt, c2_calf_opt])}")
    time_cost = time.time() - time_start
    print(f"Time Cost: {time_cost:.2f}s")
  
  elif exp_mode == "test":
    # 使用最优参数进行测试
    best_c1, best_c2 = 291.5757670770301, 98.3306672255234
    MSE_mat_overview = error([best_c1, best_c2], mode=exp_mode)
    print(MSE_mat_overview)