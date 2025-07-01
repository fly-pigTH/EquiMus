# 直接拆解数据，使用最小二乘法对模型进行参数辨识
# 产出论文数据！重点整理文件

# NOTE: can only be used for exp data in paper (first version), 
# if using the the npy data (sim with specific parameters)

import numpy as np
import math, datetime
import pandas as pd

# add exp data here
EXP_TIME = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
EXP_MODE = 'ideal_model'  # 'ideal_model' or 'real_model'
EXP_STRING = f'directOLS_[{EXP_MODE}]_[{EXP_TIME}]'

# load data
data = pd.read_csv("./src/physical_verify/static/old_data/realStaticPoint_6group.csv")
P1_array = data['P1'].values
P2_array = data['P2'].values
theta1_array = data['theta1'].values
theta2_array = data['theta2'].values

# Convert to SI units and radians
P1_actual = P1_array * 1000  # Experimental output P1 in Pa
P2_actual = P2_array * 1000  # Experimental output P2 in Pa

theta1 = theta1_array * np.pi / 180  # Input theta1 in radians
theta2 = theta2_array * np.pi / 180  # Input theta2 in radians
P2_actual.shape

# make the data from the geometry model
# For simplicity, make data without math
def get_geom_data(theta_1, theta_2):
    '''
    Args:
        theta_1: float, angle theta_1 in radians
        theta_2: float, angle theta_2 in radians

    Returns:
        Y_transpose_mat: np.ndarray, matrix of partial derivatives of y-coordinates with respect to theta_1 and theta_2
        L_transpose_mat: np.ndarray, matrix of partial derivatives of l_1 and l_2 with respect to theta_1 and theta_2
        l_1: float, length between points A and C
        l_2: float, length between points B and D
    '''
    # fixed parameters
    a_1, a_2, b_1, b_2, d_1, d_2 = 0.25, 0.25, 0.21213, 0.1, 0.06, 0.10
    beta_1, beta_2 = 8.13 / 180 * math.pi, 30 / 180 * math.pi
    g = 9.8

    A_x_O = d_1
    A_y_O = 0

    B_x_O = -d_2
    B_y_O = 0

    C_x_O = b_1 * math.cos(theta_1 - beta_1)
    C_y_O = b_1 * math.sin(theta_1 - beta_1)

    D_x_O = a_1 * math.cos(theta_1) + b_2 * math.cos(theta_1 + theta_2 + beta_2)
    D_y_O = a_1 * math.sin(theta_1) + b_2 * math.sin(theta_1 + theta_2 + beta_2)

    E_x_O = a_1 * math.cos(theta_1)
    E_y_O = a_1 * math.sin(theta_1)

    F_x_O = a_1 * math.cos(theta_1) + a_2 * math.cos(theta_1 + theta_2)
    F_y_O = a_1 * math.sin(theta_1) + a_2 * math.sin(theta_1 + theta_2)

    # Calculate partial derivatives
    # Partial derivatives with respect to theta_1

    dA_x_O_dtheta_1 = 0
    dA_y_O_dtheta_1 = 0

    dB_x_O_dtheta_1 = 0
    dB_y_O_dtheta_1 = 0

    dC_x_O_dtheta_1 = -b_1 * math.sin(theta_1 - beta_1)
    dC_y_O_dtheta_1 = b_1 * math.cos(theta_1 - beta_1)

    dD_x_O_dtheta_1 = -a_1 * math.sin(theta_1) - b_2 * math.sin(theta_1 + theta_2 + beta_2)
    dD_y_O_dtheta_1 = a_1 * math.cos(theta_1) + b_2 * math.cos(theta_1 + theta_2 + beta_2)

    dE_x_O_dtheta_1 = -a_1 * math.sin(theta_1)
    dE_y_O_dtheta_1 = a_1 * math.cos(theta_1)

    dF_x_O_dtheta_1 = -a_1 * math.sin(theta_1) - a_2 * math.sin(theta_1 + theta_2)
    dF_y_O_dtheta_1 = a_1 * math.cos(theta_1) + a_2 * math.cos(theta_1 + theta_2)

    # Partial derivatives with respect to theta_2

    dA_x_O_dtheta_2 = 0.0
    dA_y_O_dtheta_2 = 0.0

    dB_x_O_dtheta_2 = 0.0
    dB_y_O_dtheta_2 = 0.0

    dC_x_O_dtheta_2 = 0.0
    dC_y_O_dtheta_2 = 0.0

    dD_x_O_dtheta_2 = -b_2 * math.sin(theta_1 + theta_2 + beta_2)
    dD_y_O_dtheta_2 = b_2 * math.cos(theta_1 + theta_2 + beta_2)

    dE_x_O_dtheta_2 = 0.0
    dE_y_O_dtheta_2 = 0.0

    dF_x_O_dtheta_2 = -a_2 * math.sin(theta_1 + theta_2)
    dF_y_O_dtheta_2 = a_2 * math.cos(theta_1 + theta_2)

    # Calculate lengths l1 and l2
    l_1 = math.sqrt((A_x_O - C_x_O)**2 + (A_y_O - C_y_O)**2)
    l_2 = math.sqrt((B_x_O - D_x_O)**2 + (B_y_O - D_y_O)**2)

    # Calculate partial derivatives
    # Partial derivatives d/dtheta_1
    dl_1_dtheta_1 = 1/l_1 * ((A_x_O - C_x_O) * (dA_x_O_dtheta_1 - dC_x_O_dtheta_1) + (A_y_O - C_y_O) * (dA_y_O_dtheta_1 - dC_y_O_dtheta_1))
    dl_2_dtheta_1 = 1/l_2 * ((B_x_O - D_x_O) * (dB_x_O_dtheta_1 - dD_x_O_dtheta_1) + (B_y_O - D_y_O) * (dB_y_O_dtheta_1 - dD_y_O_dtheta_1))

    # Partial derivatives d/dtheta_2
    dl_1_dtheta_2 = 1/l_1 * ((A_x_O - C_x_O) * (dA_x_O_dtheta_2 - dC_x_O_dtheta_2) + (A_y_O - C_y_O) * (dA_y_O_dtheta_2 - dC_y_O_dtheta_2))
    dl_2_dtheta_2 = 1/l_2 * ((B_x_O - D_x_O) * (dB_x_O_dtheta_2 - dD_x_O_dtheta_2) + (B_y_O - D_y_O) * (dB_y_O_dtheta_2 - dD_y_O_dtheta_2))

    # Right side of the equation
    Y_transpose_mat = np.array([
        [(dE_y_O_dtheta_1/2), ((dE_y_O_dtheta_1+dF_y_O_dtheta_1)/2), (dC_y_O_dtheta_1/2), (dD_y_O_dtheta_1/2)], 
        [(dE_y_O_dtheta_2/2), ((dE_y_O_dtheta_2+dF_y_O_dtheta_2)/2), (dC_y_O_dtheta_2/2), (dD_y_O_dtheta_2/2)]])

    # Left side of the equation
    L_transpose_mat = np.array([
        [dl_1_dtheta_1, dl_2_dtheta_1],
        [dl_1_dtheta_2, dl_2_dtheta_2]])

    return Y_transpose_mat, L_transpose_mat, l_1, l_2


# Construct the least squares problem (auxiliary variable trick) - pneumatic model
# Variable set: S1, S2, [m1, m2, m3, m4], k1, k2, k1*l10, k2*l20
var_num = 6
data_size = len(theta1)
A_mat = np.zeros((data_size*2, var_num))
b_mat = np.zeros((data_size*2, 1))

# NOTE: directly use the average of the actuators
m1, m2, m3, m4 = 0.086, 0.1033, 186.48 * 1e-3, 272.66 * 1e-3
g = 9.8
m_array = np.array([m1, m2, m3, m4]).reshape(-1, 1)

# minist_square_problem:
# x = [1/s1, 1/s2, k1/s1, k2/s2, k1*l10/s1, k2*l20/s2].T
# therefore A = [[-G[0], l1, -1, -G[1], l2, -1], ...]
# b = [P1, P2].T

for i in range(data_size):
    Y_transpose_mat, L_transpose_mat, l_1, l_2 = get_geom_data(theta1[i], theta2[i])

    minus_g_mat = -g*np.linalg.inv(L_transpose_mat)@Y_transpose_mat@m_array

    A_mat[i*2, 0] = minus_g_mat[0, 0]
    A_mat[i*2, 1] = 0
    A_mat[i*2, 2] = l_1
    A_mat[i*2, 3] = 0
    A_mat[i*2, 4] = -1
    A_mat[i*2, 5] = 0

    A_mat[i*2+1, 0] = 0
    A_mat[i*2+1, 1] = minus_g_mat[1, 0]
    A_mat[i*2+1, 2] = 0
    A_mat[i*2+1, 3] = l_2
    A_mat[i*2+1, 4] = 0
    A_mat[i*2+1, 5] = -1

    b_mat[i*2, 0] = P1_actual[i]
    b_mat[i*2+1, 0] = P2_actual[i]

# solve Ax = b
solution = np.linalg.inv(A_mat.T@A_mat)@A_mat.T@b_mat
s1_prime = 1/solution[0, 0]
s2_prime = 1/solution[1, 0]
k1_prime = solution[2, 0]*s1_prime
k2_prime = solution[3, 0]*s2_prime
l10_prime = solution[4, 0]/k1_prime*s1_prime
l20_prime = solution[5, 0]/k2_prime*s2_prime

# print the result
print(f"s1: {s1_prime}, s2: {s2_prime}, \nk1: {k1_prime}, k2: {k2_prime}, \nl10: {l10_prime}, l20: {l20_prime}")
# save in csv vs pandas
prime_parameter = pd.DataFrame({
    's1': [s1_prime],
    's2': [s2_prime],
    'k1': [k1_prime],
    'k2': [k2_prime],
    'l10': [l10_prime],
    'l20': [l20_prime]
})
prime_parameter.to_csv(f"./src/physical_verify/static/old_log/static_parameter_{EXP_STRING}.csv", index=False)

# real vs sim(ideal)

# real theta
theta1_real = theta1
theta2_real = theta2

if EXP_MODE == "ideal_model":
    _sim_data = np.load("./src/physical_verify/static/old_data/simExp-sim-real constrast-20250219_230253/StaticState_list.npy")        # the ideal model
else:
    _sim_data = np.load("./src/physical_verify/static/old_data/simExp-sim-real constrast-20250219_231642/StaticState_list.npy")        # the geom model
# TODO: 这里读取的 sim 数据是直接跑好的，未来改为程序中生成，或者保存后从中间结果 (csv) 中读取
theta_sim = _sim_data[:, 3:5]
theta1_sim = theta_sim[:, 0] + math.pi/2
theta2_sim = theta_sim[:, 1]
# for geom model
if EXP_MODE == "real_model":
    theta2_sim += math.pi/2

# cal error
theta1_error = theta1_real - theta1_sim
theta2_error = theta2_real - theta2_sim
theta1_relative_error = theta1_error / theta1_real
theta2_relative_error = theta2_error / theta2_real

static_error_summary_data = {
    'Pressure1/kPa': P1_actual/1000,
    'Pressure2/kPa': P2_actual/1000,
    'Theta1_real/deg': theta1_real/math.pi*180,
    'Theta1_sim/deg': theta1_sim/math.pi*180,
    'Theta1_error/deg': theta1_error/math.pi*180,
    'Theta1_errorRelative': theta1_relative_error,

    'Theta2_real/deg': theta2_real/math.pi*180,
    'Theta2_sim/deg': theta2_sim/math.pi*180,
    'Theta2_error/deg': theta2_error/math.pi*180,
    'Theta2_errorRelative': theta2_relative_error
}
static_error_summary_data = pd.DataFrame(static_error_summary_data)
static_error_summary_data.to_csv(f"./src/physical_verify/static/old_log/static_error_summary_data_{EXP_STRING}.csv", index=False)

# cal MaxAE, MAE, RMSE
theta1_MaxAE = np.max(np.abs(theta1_error))
theta1_MAE = np.mean(np.abs(theta1_error))
theta1_RMSE = np.sqrt(np.mean(theta1_error**2))

theta2_MaxAE = np.max(np.abs(theta2_error))
theta2_MAE = np.mean(np.abs(theta2_error))
theta2_RMSE = np.sqrt(np.mean(theta2_error**2))

# save the data
error_theta_summary = pd.DataFrame({
    'Theta1_MaxAE/deg': [theta1_MaxAE/math.pi*180],
    'Theta1_MAE/deg': [theta1_MAE/math.pi*180],
    'Theta1_RMSE/deg': [theta1_RMSE/math.pi*180],
    'Theta2_MaxAE/deg': [theta2_MaxAE/math.pi*180],
    'Theta2_MAE/deg': [theta2_MAE/math.pi*180],
    'Theta2_RMSE/deg': [theta2_RMSE/math.pi*180]
})
error_theta_summary.to_csv(f"./src/physical_verify/static/old_log/error_theta_summary_{EXP_STRING}.csv", index=False)

# show the result
print("static_error_summary_data")
print(static_error_summary_data)

print("error_theta_summary")
print(error_theta_summary)
