'''
    给转动关节角度，给出关键点坐标，计算完整的qpos

'''
import numpy as np
import math

def qpos_cal(theta_1, theta_2, beta_1=8.13/180*math.pi, beta_2=30/180*math.pi, d_1=0.06, d_2=0.10, a_1=0.25, a_2=0.25, b_1=0.21213, b_2=0.10):
    '''
        input: theta_1, theta_2
        output: qpos[theta_1, theta_2, uang_1, uang_2, l_1, l_2]
    '''
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

    O_array = np.array([O_x_O, O_y_O])
    A_array = np.array([A_x_O, A_y_O])
    B_array = np.array([B_x_O, B_y_O])
    C_array = np.array([C_x_O, C_y_O])
    D_array = np.array([D_x_O, D_y_O])

    l_1 = np.linalg.norm(C_array - A_array)
    l_2 = np.linalg.norm(D_array - B_array)
    
    uang_1 = math.atan2(C_y_O - A_y_O, C_x_O - A_x_O)
    uang_2 = math.atan2(D_y_O - B_y_O, D_x_O - B_x_O)

    qpos = np.array([theta_1, theta_2, uang_1, uang_2, l_1, l_2])
    return qpos


if __name__ == '__main__':
    theta_1 = 90*math.pi/180
    theta_2 = 0
    qpos = qpos_cal(theta_1, theta_2)
    print(qpos)
