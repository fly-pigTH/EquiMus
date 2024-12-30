# landing model 2DOF 静力学推导，考察系统非稳态？

import numpy as np
import math
import scipy.io

theta_1 = 1.067# 1.4681013268 #1.498 # math.pi/2
theta_2 = 1.14 # 0.468562#0.788 # math.pi/4

a_1 = 0.25
a_2 = 0.25
b_1 = 0.21213
b_2 = 0.1
d_1 = 0.06
d_2 = 0.10
s = 0

M = 0.155  # 真实值
m_1, m_2 = 0.086, 0.1033
I_1, I_2 = (1/12 * m_1 * 0.25**2), (1/12 * m_2 * 0.25**2)

m_3, m_4 = 0.18, 0.18

# 刚度参数
k_3, k_4 = 637.52/2, 631.6/2

# 阻尼参数
c_3, c_4 = 22.68/2, 21.8/2

# 重力加速度
g = 0 #9.8

# 角度参数 (度转弧度)
beta_1 = 8.13 / 180 * np.pi
beta_2 = 30 / 180 * np.pi

# 长度参数
l_10 = 0.174
l_20 = 0.252

O_x_O = 0
O_y_O = 0

A_x_O = d_1
A_y_O = 0

B_x_O = -d_2
B_y_O = 0

C_x_O = b_1 * np.cos(theta_1 - beta_1)
C_y_O = b_1 * np.sin(theta_1 - beta_1)

D_x_O = a_1 * np.cos(theta_1) + b_2 * np.cos(theta_1 + theta_2 + beta_2)
D_y_O = a_1 * np.sin(theta_1) + b_2 * np.sin(theta_1 + theta_2 + beta_2)

E_x_O = a_1 * np.cos(theta_1)
E_y_O = a_1 * np.sin(theta_1)

F_x_O = a_1 * np.cos(theta_1) + a_2 * np.cos(theta_1 + theta_2)
F_y_O = a_1 * np.sin(theta_1) + a_2 * np.sin(theta_1 + theta_2)

# G_x_O = b_1 * np.cos(beta_1) * np.cos(theta_1)
# G_y_O = b_1 * np.cos(beta_1) * np.sin(theta_1)

# H_x_O = a_1 * np.cos(theta_1) + b_2 * np.cos(beta_2) * np.cos(theta_1 + theta_2)
# H_y_O = a_1 * np.sin(theta_1) + b_2 * np.cos(beta_2) * np.sin(theta_1 + theta_2)

# 计算偏导数
# 对 theta_1 的偏导数
dO_x_O_dtheta_1 = 0
dO_y_O_dtheta_1 = 0

dA_x_O_dtheta_1 = 0
dA_y_O_dtheta_1 = 0

dB_x_O_dtheta_1 = 0
dB_y_O_dtheta_1 = 0

dC_x_O_dtheta_1 = -b_1 * np.sin(theta_1 - beta_1)
dC_y_O_dtheta_1 = b_1 * np.cos(theta_1 - beta_1)

dD_x_O_dtheta_1 = -a_1 * np.sin(theta_1) - b_2 * np.sin(theta_1 + theta_2 + beta_2)
dD_y_O_dtheta_1 = a_1 * np.cos(theta_1) + b_2 * np.cos(theta_1 + theta_2 + beta_2)

dE_x_O_dtheta_1 = -a_1 * np.sin(theta_1)
dE_y_O_dtheta_1 = a_1 * np.cos(theta_1)

dF_x_O_dtheta_1 = -a_1 * np.sin(theta_1) - a_2 * np.sin(theta_1 + theta_2)
dF_y_O_dtheta_1 = a_1 * np.cos(theta_1) + a_2 * np.cos(theta_1 + theta_2)

# dG_x_O_dtheta_1 = -b_1 * np.cos(beta_1) * np.sin(theta_1)
# dG_y_O_dtheta_1 = b_1 * np.cos(beta_1) * np.cos(theta_1)

# dH_x_O_dtheta_1 = -a_1 * np.sin(theta_1) - b_2 * np.cos(beta_2) * np.sin(theta_1 + theta_2)
# dH_y_O_dtheta_1 = a_1 * np.cos(theta_1) + b_2 * np.cos(beta_2) * np.cos(theta_1 + theta_2)

# 对 theta_2 的偏导数

dO_x_O_dtheta_2 = 0
dO_y_O_dtheta_2 = 0

dA_x_O_dtheta_2 = 0
dA_y_O_dtheta_2 = 0

dB_x_O_dtheta_2 = 0
dB_y_O_dtheta_2 = 0

dC_x_O_dtheta_2 = 0
dC_y_O_dtheta_2 = 0

dD_x_O_dtheta_2 = -b_2 * np.sin(theta_1 + theta_2 + beta_2)
dD_y_O_dtheta_2 = b_2 * np.cos(theta_1 + theta_2 + beta_2)

dE_x_O_dtheta_2 = 0
dE_y_O_dtheta_2 = 0

dF_x_O_dtheta_2 = -a_2 * np.sin(theta_1 + theta_2)
dF_y_O_dtheta_2 = a_2 * np.cos(theta_1 + theta_2)

# dH_x_O_dtheta_2 = -b_2 * np.cos(beta_2) * np.sin(theta_1 + theta_2)
# dH_y_O_dtheta_2 = b_2 * np.cos(beta_2) * np.cos(theta_1 + theta_2)


# 把以上所有参数都减掉F
O_x_O = O_x_O - F_x_O
O_y_O = O_y_O - F_y_O
A_x_O = A_x_O - F_x_O
A_y_O = A_y_O - F_y_O
B_x_O = B_x_O - F_x_O
B_y_O = B_y_O - F_y_O
C_x_O = C_x_O - F_x_O
C_y_O = C_y_O - F_y_O
D_x_O = D_x_O - F_x_O
D_y_O = D_y_O - F_y_O
E_x_O = E_x_O - F_x_O
E_y_O = E_y_O - F_y_O
# G_x_O = G_x_O - F_x_O
# G_y_O = G_y_O - F_y_O
# H_x_O = H_x_O - F_x_O
# H_y_O = H_y_O - F_y_O

F_x_O -= F_x_O
F_y_O -= F_y_O

# A 点
dO_x_O_dtheta_1 -= dF_x_O_dtheta_1
dO_y_O_dtheta_1 -= dF_y_O_dtheta_1
dO_x_O_dtheta_2 -= dF_x_O_dtheta_2
dO_y_O_dtheta_2 -= dF_y_O_dtheta_2

# A 点
dA_x_O_dtheta_1 -= dF_x_O_dtheta_1
dA_y_O_dtheta_1 -= dF_y_O_dtheta_1
dA_x_O_dtheta_2 -= dF_x_O_dtheta_2
dA_y_O_dtheta_2 -= dF_y_O_dtheta_2

# B 点
dB_x_O_dtheta_1 -= dF_x_O_dtheta_1
dB_y_O_dtheta_1 -= dF_y_O_dtheta_1
dB_x_O_dtheta_2 -= dF_x_O_dtheta_2
dB_y_O_dtheta_2 -= dF_y_O_dtheta_2

# C 点
dC_x_O_dtheta_1 -= dF_x_O_dtheta_1
dC_y_O_dtheta_1 -= dF_y_O_dtheta_1
dC_x_O_dtheta_2 -= dF_x_O_dtheta_2
dC_y_O_dtheta_2 -= dF_y_O_dtheta_2


# D 点
dD_x_O_dtheta_1 -= dF_x_O_dtheta_1
dD_y_O_dtheta_1 -= dF_y_O_dtheta_1
dD_x_O_dtheta_2 -= dF_x_O_dtheta_2
dD_y_O_dtheta_2 -= dF_y_O_dtheta_2

# E 点
dE_x_O_dtheta_1 -= dF_x_O_dtheta_1
dE_y_O_dtheta_1 -= dF_y_O_dtheta_1
dE_x_O_dtheta_2 -= dF_x_O_dtheta_2
dE_y_O_dtheta_2 -= dF_y_O_dtheta_2

# # G 点
# dG_x_O_dtheta_1 -= dF_x_O_dtheta_1
# dG_y_O_dtheta_1 -= dF_y_O_dtheta_1
# dG_x_O_dtheta_2 -= dF_x_O_dtheta_2
# dG_y_O_dtheta_2 -= dF_y_O_dtheta_2

# # H 点
# dH_x_O_dtheta_1 -= dF_x_O_dtheta_1
# dH_y_O_dtheta_1 -= dF_y_O_dtheta_1
# dH_x_O_dtheta_2 -= dF_x_O_dtheta_2
# dH_y_O_dtheta_2 -= dF_y_O_dtheta_2

# F
dF_x_O_dtheta_1 -= dF_x_O_dtheta_1
dF_y_O_dtheta_1 -= dF_y_O_dtheta_1
dF_x_O_dtheta_2 -= dF_x_O_dtheta_2
dF_y_O_dtheta_2 -= dF_y_O_dtheta_2



# 计算长度 l1 和 l2
l_1 = np.sqrt((A_x_O - C_x_O)**2 + (A_y_O - C_y_O)**2)
l_2 = np.sqrt((B_x_O - D_x_O)**2 + (B_y_O - D_y_O)**2)

# 计算偏导数
# 偏导数 d/dtheta_1
dl_1_dtheta_1 = 1/l_1 * ((A_x_O - C_x_O) * (dA_x_O_dtheta_1 - dC_x_O_dtheta_1) + (A_y_O - C_y_O) * (dA_y_O_dtheta_1 - dC_y_O_dtheta_1))
dl_2_dtheta_1 = 1/l_2 * ((B_x_O - D_x_O) * (dB_x_O_dtheta_1 - dD_x_O_dtheta_1) + (B_y_O - D_y_O) * (dB_y_O_dtheta_1 - dD_y_O_dtheta_1))

# 偏导数 d/dtheta_2
dl_1_dtheta_2 = 1/l_1 * ((A_x_O - C_x_O) * (dA_x_O_dtheta_2 - dC_x_O_dtheta_2) + (A_y_O - C_y_O) * (dA_y_O_dtheta_2 - dC_y_O_dtheta_2))
dl_2_dtheta_2 = 1/l_2 * ((B_x_O - D_x_O) * (dB_x_O_dtheta_2 - dD_x_O_dtheta_2) + (B_y_O - D_y_O) * (dB_y_O_dtheta_2 - dD_y_O_dtheta_2))

print(dl_1_dtheta_2)

# 等式右侧
RHSb_1 = -( m_1*g*((dE_y_O_dtheta_1+dO_y_O_dtheta_1)/2) + m_2*g*((dE_y_O_dtheta_1+dF_y_O_dtheta_1)/2) + m_3*g*((dA_y_O_dtheta_1+dC_y_O_dtheta_1)/2) + m_4*g*((dB_y_O_dtheta_1+dD_y_O_dtheta_1)/2) + M*g*dO_y_O_dtheta_1)
RHSb_2 = -( m_1*g*((dE_y_O_dtheta_2+dO_y_O_dtheta_2)/2) + m_2*g*((dE_y_O_dtheta_2+dF_y_O_dtheta_2)/2) + m_3*g*((dA_y_O_dtheta_2+dC_y_O_dtheta_2)/2) + m_4*g*((dB_y_O_dtheta_2+dD_y_O_dtheta_2)/2) + M*g*dO_y_O_dtheta_2)
b = np.array([RHSb_1, RHSb_2])
print(f"b: {b}")

# 等式左侧
LHSA = np.array([[dl_1_dtheta_1, dl_2_dtheta_1], 
                [dl_1_dtheta_2, dl_2_dtheta_2]])
print(f"LHSA: {LHSA}")

# 回复力
F_k = np.array([k_3*(l_1-l_10), k_4*(l_2-l_20)])

# 计算静力学
StaticForce = np.linalg.solve(LHSA, b) + F_k
print("StaticForce", StaticForce)

dF_y_O_dtheta_1, dF_y_O_dtheta_2

