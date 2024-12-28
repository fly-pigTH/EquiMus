import mujoco
import numpy as np
import matplotlib.pyplot as plt

# 加载模型
m = mujoco.MjModel.from_xml_path("parallel.xml")
d = mujoco.MjData(m)

# 模拟时间
time = 0
frequency = 0.25  # 目标正弦运动的频率
amplitude = 0.5  # 最大控制角度
ang_list = []
t_list = []

with mujoco.viewer.launch_passive(m, d) as viewer:
    d_startTime = d.time
    while True and d.time-d_startTime < 10:
        time = d.time - d_startTime
        t_list.append(d.time)
        ang_list.append(d.qpos[0])
        # 计算目标关节状态
        desired_qpos = amplitude * np.sin(2 * np.pi * frequency * time)
        desired_qvel = 2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * time)
        desired_qacc = -(2 * np.pi * frequency)**2 * amplitude * np.sin(2 * np.pi * frequency * time)

        # 设置期望状态到仿真器
        # d.qpos[0] = desired_qpos
        # d.qvel[0] = desired_qvel
        d.qacc[0] = desired_qacc

        # 调用逆动力学
        mujoco.mj_inverse(m, d)

        # 获取计算的力矩
        torque = d.qfrc_inverse
        # print(torque)
        torque0 = torque[0]

        # 施加到关节 j0
        d.ctrl[0] = torque0
        print(d.ctrl)

        # 仿真一步并渲染
        mujoco.mj_step(m, d)

        viewer.sync()

# save the data
np.save("t_list.npy", t_list)
np.save("ang_list.npy", ang_list)