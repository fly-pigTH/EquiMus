'''
    为静力学模型设置PID控制器
    计算静力学模型
'''
import time
import mujoco
import mujoco.viewer
import numpy as np
import random
import math
from qpos_cal import qpos_cal

path2 = '/Users/flypig/Documents/Coding/MujocoLearn/v2_4/urdf/dog2_4singleLeg.xml'
m = mujoco.MjModel.from_xml_path(path2)
d = mujoco.MjData(m)

# init
print(m.body_mass)
mujoco.mj_forward(m, d)

# 获取当前状态
print("qpos:", d.qpos)
print("qvel:", d.qvel)
print("qacc:", d.qacc)
print("----")

# Check the inverse dynamics
duration = 200
start = time.time()

# 设置 qpos6-RB_BAA_Slide 为 0
target_id = [3, 6]
target_pos = np.zeros_like(target_id, dtype=np.float64)
error_integral = np.zeros_like(target_id, dtype=np.float64)
error_last = np.zeros_like(target_id, dtype=np.float64)
k_p = 1000
k_i = 10
k_d = 0.3

with mujoco.viewer.launch_passive(m, d) as viewer:
    # 30时间步长后关闭viewer
    start = time.time()
    while viewer.is_running() and time.time() - start < duration:
        step_start = time.time()

        error = target_pos - d.qpos[target_id]
        error_integral += error
        error_derivative = error - error_last

        # set the controller
        F_kp = k_p * error
        F_ki = k_i * error_integral
        F_kd = k_d * error_derivative
        print(f"[F_kp]: {F_kp}, [F_ki]: {F_ki}, [F_kd]: {F_kd}")
        d.ctrl[[1, 3]] = k_p * error + k_i * error_integral + k_d * error_derivative
        mujoco.mj_step(m, d)    # forward dynamics!

        error_last = error

        # print(f"d.qpos: {d.qpos}")

        viewer.sync()

        # 粗略的计时，相对于挂钟会有漂移。
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("Finish!")

# record the last state
print("qpos:", d.qpos)

last_qpos = d.qpos

# 重置状态
d.qpos[:] = last_qpos
# give a disturbance
d.qpos[-1] += 0.00001
d.qvel[:] = 0  # 所有关节的速度为零
d.qacc[:] = 0  # 所有关节的加速度为零

# 执行逆向动力学
mujoco.mj_inverse(m, d)

static_force_last_state = d.qfrc_inverse

# print("qpos:", d.qpos)
# print("qvel:", d.qvel)
# print("qacc:", d.qacc)
# print("ctrl:", d.ctrl)

print(f"inverse(static)-using sim-qpos: {d.qfrc_inverse}")
print(f"end state: {d.qpos}")


# qpos计算check

# 初始长度和uang需要单独处理
init_qpos = qpos_cal(90*math.pi/180, 90*math.pi/180)
pres_qpos = qpos_cal(90*math.pi/180+d.qpos[0], 90*math.pi/180+d.qpos[1])
delta_qpos = pres_qpos - init_qpos 
print(f"delta_qpos: {delta_qpos}")
