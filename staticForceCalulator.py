import time
import mujoco
import mujoco.viewer
import numpy as np
import random
import math
from qpos_cal import qpos_cal

path1 = "/Users/flypig/Documents/Coding/MujocoLearn/1101/1101/mydog110.xml"
path2 = '/Users/flypig/Documents/Coding/MujocoLearn/v2_4/urdf/dog2_4singleLeg.xml'
path3 = '/Users/flypig/Documents/Coding/MujocoLearn/v2_4/urdf/dog2_4.xml'
m = mujoco.MjModel.from_xml_path(path2)
d = mujoco.MjData(m)
flag = 0

# init
print(m.body_mass)

mujoco.mj_forward(m, d)

# 获取当前状态
print("qpos:", d.qpos)
print("qvel:", d.qvel)
print("qacc:", d.qacc)
print("----")

# Check the inverse dynamics
duration = 20
start = time.time()
if True:
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 30时间步长后关闭viewer
        start = time.time()
        while viewer.is_running() and time.time() - start < duration:
            step_start = time.time()

            d.ctrl[0] = 1.14514
            d.ctrl[1] = 1.14514*(-1)
            d.ctrl[2] = 1.14514*(0.1)
            d.ctrl[3] = 1.14514*(10)

            mujoco.mj_step(m, d)    # forward dynamics!

            print(f"d.qpos: {d.qpos}")

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

print('CHECK')
delta_unag1 = delta_qpos[2] - d.qpos[2]
delta_unag2 = delta_qpos[3] - d.qpos[5]      # NOTE: 由于qpos[5]是反的
delta_l1 = delta_qpos[4] - (d.qpos[3]+d.qpos[4])
delta_l2 = delta_qpos[5] - (d.qpos[6]+d.qpos[6])
print(f"delta_unag1: {delta_unag1}")
print(f"delta_unag2: {delta_unag2}")
print(f"delta_l1: {delta_l1}")
print(f"delta_l2: {delta_l2}")
input()

## Check the static force of real qpos and cal qpos

print(f"inverse(static)-using real-qpos: {static_force_last_state}")

# 计算真实cal qpos对应的静力学
# 重置状态
d.qpos[:] = 0
d.qpos[:2] = delta_qpos[:2]
d.qpos[2] = delta_qpos[2]
d.qpos[5] = delta_qpos[3]
d.qpos[3] = delta_qpos[4]/2
d.qpos[4] = delta_qpos[4]/2
d.qpos[6] = delta_qpos[5]/2
d.qpos[7] = delta_qpos[5]/2-0.0001
d.qvel[:] = 0  # 所有关节的速度为零
d.qacc[:] = 0  # 所有关节的加速度为零

# 执行逆向动力学
mujoco.mj_inverse(m, d)
static_force = d.qfrc_inverse
print(f"inverse(static)-using cal-qpos: {d.qfrc_inverse}")

# 重置状态. 对比！
d.qpos[:] = 0
d.qpos[[3, 4, 6, 7]] = last_qpos[[3, 4, 6, 7]]
d.qvel[:] = 0  # 所有关节的速度为零
d.qacc[:] = 0  # 所有关节的加速度为零

# 执行逆向动力学
mujoco.mj_inverse(m, d)

# print("qpos:", d.qpos)
# print("qvel:", d.qvel)
# print("qacc:", d.qacc)
# print("ctrl:", d.ctrl)

print(f"inverse(static)-using direct-qpos: {d.qfrc_inverse}")


print("---")
input("---STOP HERE---")


## 运动学约束设置
# 重置状态
d.qpos[:] = 0
d.qvel[:] = 0  # 所有关节的速度为零
d.qacc[:] = 0  # 所有关节的加速度为零
d.ctrl[:] = 0  # 所有关节的控制信号为零

# 设置主动关节位置
d.qpos[3] = 0.2  # 假设设置第一个关节的位置为 0.2（单位视模型而定）
d.qpos[4] = 0.2  # 假设设置第二个关节的位置为 -0.2（单位视模型而定）
d.qpos[6] = 0.2  # 假设设置第一个关节的位置为 0.2（单位视模型而定）
d.qpos[7] = 0.2  # 假设设置第二个关节的位置为 -0.2（单位视模型而定）

# 固定速度
d.qvel[3] = 0  # 假设设置第一个关节的速度为 0
d.qvel[4] = 0  # 假设设置第二个关节的速度为 0
d.qvel[6] = 0  # 假设设置第一个关节的速度为 0
d.qvel[7] = 0  # 假设设置第二个关节的速度为 0

# trick: 通过静力学给一个“虚拟”的力
# 执行逆向动力学

# mujoco.mj_activate(m)

mujoco.mj_forward(m, d)
print(f"d.qpos: {d.qpos}")


# 30时间步长后关闭viewer
duration = 100000
round = 0
start = time.time()
with mujoco.viewer.launch_passive(m, d) as viewer:
    # 30时间步长后关闭viewer
    start = time.time()
    while viewer.is_running() and time.time() - start < duration:
        step_start = time.time()

        # d.ctrl[0] = 1.14514
        # d.ctrl[1] = 1.14514*(-1)
        # d.ctrl[2] = 1.14514*(0.1)
        # d.ctrl[3] = 1.14514*(10)

        d.qpos[3] = 0.02  # 假设设置第一个关节的位置为 0.2（单位视模型而定）
        d.qpos[4] = 0.02  # 假设设置第二个关节的位置为 -0.2（单位视模型而定）
        d.qpos[6] = 0.05  # 假设设置第一个关节的位置为 0.2（单位视模型而定）
        d.qpos[7] = 0.05  # 假设设置第二个关节的位置为 -0.2（单位视模型而定）

        # 固定速度
        d.qvel[3] = 0  # 假设设置第一个关节的速度为 0
        d.qvel[4] = 0  # 假设设置第二个关节的速度为 0
        d.qvel[6] = 0  # 假设设置第一个关节的速度为 0
        d.qvel[7] = 0  # 假设设置第二个关节的速度为 0

        mujoco.mj_step(m, d)    # forward dynamics!

        print(f"d.qpos: {d.qpos}")

        viewer.sync()

        if round < 5:
            input()

        # 粗略的计时，相对于挂钟会有漂移。
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        round += 1

print("qpos:", d.qpos)