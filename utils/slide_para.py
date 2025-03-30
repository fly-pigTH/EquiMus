import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 假设这是你的 run_single_experiment 函数
def run_single_experiment(para, freq1, freq2, save=False):
    # 这里用随机数据模拟仿真结果
    time_sim = np.linspace(0, 4, 1000)
    theta1_sim = np.sin(2 * np.pi * freq1 * time_sim) * para[4]  # 模拟 theta1
    theta2_sim = np.cos(2 * np.pi * freq2 * time_sim) * para[5]  # 模拟 theta2
    return time_sim, theta1_sim, theta2_sim

# 初始参数
k1 = 130.4233
k2 = 117.9855
l10 = 0.1640
l20 = 0.2580
c1_init = 99.6650  # 初始 c1 值
c2_init = 42.6441  # 初始 c2 值
s1 = 0.00039486
s2 = 0.00044950

# 初始参数列表
para = [k1, k2, l10, l20, c1_init, c2_init, s1, s2]

# 模拟实验数据
time_sim, theta1_sim, theta2_sim = run_single_experiment(para, 30 * 10**3, 10 * 10**3, save=False)

# 真实实验数据（假设 expdata_real_all 是一个包含时间、theta1、theta2 的列表）
expdata_real_all = {
    19: [
        np.linspace(0, 4, 1000),  # 时间
        np.sin(2 * np.pi * 30 * 10**3 * np.linspace(0, 4, 1000)),  # theta1_real
        np.cos(2 * np.pi * 10 * 10**3 * np.linspace(0, 4, 1000)),  # theta2_real
    ]
}

# 创建图形和轴
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.35)  # 调整布局，为滑动杆留出空间

# 绘制初始仿真曲线
l_sim1, = plt.plot(time_sim, theta1_sim, label="theta1_sim")
# l_sim2, = plt.plot(time_sim, theta2_sim, label="theta2_sim")

# 绘制真实数据曲线
l_real1, = plt.plot(expdata_real_all[19][0], expdata_real_all[19][1], label="theta1_real")
# l_real2, = plt.plot(expdata_real_all[19][0], expdata_real_all[19][2], label="theta2_real")

plt.legend()
plt.xlim(0, 4)
plt.ylim(-150, 150)  # 设置 y 轴范围

# 创建滑动杆轴
ax_c1 = plt.axes([0.25, 0.15, 0.65, 0.03])  # c1 滑动杆位置
ax_c2 = plt.axes([0.25, 0.1, 0.65, 0.03])   # c2 滑动杆位置

# 创建滑动杆
c1_slider = Slider(ax_c1, 'c1', 0, 200, valinit=c1_init)
c2_slider = Slider(ax_c2, 'c2', 0, 100, valinit=c2_init)

# 更新函数
def update(val):
    # 获取滑动杆的值
    c1 = c1_slider.val
    c2 = c2_slider.val

    # 更新参数列表
    para[4] = c1
    para[5] = c2

    # 重新运行仿真
    time_sim, theta1_sim, theta2_sim = run_single_experiment(para, 30 * 10**3, 10 * 10**3, save=False)

    # 更新曲线数据
    l_sim1.set_ydata(theta1_sim)
    # l_sim2.set_ydata(theta2_sim)

    # 重绘图
    fig.canvas.draw_idle()

# 绑定滑动杆事件
c1_slider.on_changed(update)
c2_slider.on_changed(update)

# 显示图形
plt.show()