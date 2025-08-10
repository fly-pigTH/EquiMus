# Simulated Verificaition, dynamics with STANCE PHASE
# This scripts the swing trajectory for the simulated robot, plan scan (10, 10) x (10, 10) grid

import numpy as np
import math
from tqdm import tqdm
import pandas as pd
import mediapy as media

import rootpath
import sys
from pathlib import Path

ROOT_DIR = rootpath.detect()
sys.path.append(str(Path(ROOT_DIR)))
from utils.experiment import MujocoExperiment

CURRENT_DIR = Path(__file__).resolve().parent
EXP_DIR = CURRENT_DIR.parent
VIDEO_DIR = EXP_DIR / "video"
DATA_DIR = EXP_DIR / "data"

# Static force calculation module
# Need change the parameters here!
def get_static_func(theta_1, theta_2):

    a_1 = 0.25
    a_2 = 0.25
    b_1 = 0.21213
    b_2 = 0.1
    d_1 = 0.06
    d_2 = 0.10
    s = 0

    M = 0.155  # real value
    m_1, m_2 = 0.086, 0.1033
    I_1, I_2 = (1/12 * m_1 * 0.25**2), (1/12 * m_2 * 0.25**2)

    # NOTE: Use measured value here
    m_3, m_4 = 0.18648, 0.27266
    k_3, k_4 = 637.52/2, 631.6/2
    l_10, l_20 = 0.174, 0.2562

    # damping
    c_3, c_4 = 22.68/2, 21.8/2

    g = 9.8
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

    E_x_O = a_1 * np.cos(theta_1)
    E_y_O = a_1 * np.sin(theta_1)

    F_x_O = a_1 * np.cos(theta_1) + a_2 * np.cos(theta_1 + theta_2)
    F_y_O = a_1 * np.sin(theta_1) + a_2 * np.sin(theta_1 + theta_2)

    G_x_O = b_1 * np.cos(beta_1) * np.cos(theta_1)
    G_y_O = b_1 * np.cos(beta_1) * np.sin(theta_1)

    H_x_O = a_1 * np.cos(theta_1) + b_2 * np.cos(beta_2) * np.cos(theta_1 + theta_2)
    H_y_O = a_1 * np.sin(theta_1) + b_2 * np.cos(beta_2) * np.sin(theta_1 + theta_2)

    # Compute partial derivatives
    # Partial derivatives with respect to theta_1

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

    dG_x_O_dtheta_1 = -b_1 * np.cos(beta_1) * np.sin(theta_1)
    dG_y_O_dtheta_1 = b_1 * np.cos(beta_1) * np.cos(theta_1)

    dH_x_O_dtheta_1 = -a_1 * np.sin(theta_1) - b_2 * np.cos(beta_2) * np.sin(theta_1 + theta_2)
    dH_y_O_dtheta_1 = a_1 * np.cos(theta_1) + b_2 * np.cos(beta_2) * np.cos(theta_1 + theta_2)

    # Partial derivatives with respect to theta_1

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

    dH_x_O_dtheta_2 = -b_2 * np.cos(beta_2) * np.sin(theta_1 + theta_2)
    dH_y_O_dtheta_2 = b_2 * np.cos(beta_2) * np.cos(theta_1 + theta_2)


    # calcuate l1 and l2
    l_1 = np.sqrt((A_x_O - C_x_O)**2 + (A_y_O - C_y_O)**2)
    l_2 = np.sqrt((B_x_O - D_x_O)**2 + (B_y_O - D_y_O)**2)

    # partial dl/dtheta_1
    dl_1_dtheta_1 = 1/l_1 * ((A_x_O - C_x_O) * (dA_x_O_dtheta_1 - dC_x_O_dtheta_1) + (A_y_O - C_y_O) * (dA_y_O_dtheta_1 - dC_y_O_dtheta_1))
    dl_2_dtheta_1 = 1/l_2 * ((B_x_O - D_x_O) * (dB_x_O_dtheta_1 - dD_x_O_dtheta_1) + (B_y_O - D_y_O) * (dB_y_O_dtheta_1 - dD_y_O_dtheta_1))

    # partial dl/dtheta_2
    dl_1_dtheta_2 = 1/l_1 * ((A_x_O - C_x_O) * (dA_x_O_dtheta_2 - dC_x_O_dtheta_2) + (A_y_O - C_y_O) * (dA_y_O_dtheta_2 - dC_y_O_dtheta_2))
    dl_2_dtheta_2 = 1/l_2 * ((B_x_O - D_x_O) * (dB_x_O_dtheta_2 - dD_x_O_dtheta_2) + (B_y_O - D_y_O) * (dB_y_O_dtheta_2 - dD_y_O_dtheta_2))

    # print(dl_1_dtheta_2)  # check the model

    # RHS
    RHSb_1 = -( m_1*g*(dE_y_O_dtheta_1/2) + m_2*g*((dE_y_O_dtheta_1+dF_y_O_dtheta_1)/2) + m_3*g*(dC_y_O_dtheta_1/2) + m_4*g*(dD_y_O_dtheta_1/2) )
    RHSb_2 = -( m_1*g*(dE_y_O_dtheta_2/2) + m_2*g*((dE_y_O_dtheta_2+dF_y_O_dtheta_2)/2) + m_3*g*(dC_y_O_dtheta_2/2) + m_4*g*(dD_y_O_dtheta_2/2) )
    b = np.array([RHSb_1, RHSb_2])

    # LHS
    LHSA = np.array([[dl_1_dtheta_1, dl_2_dtheta_1], 
                    [dl_1_dtheta_2, dl_2_dtheta_2]])

    # recovery force
    F_k = np.array([k_3*(l_1-l_10), k_4*(l_2-l_20)])

    StaticForce = np.linalg.solve(LHSA, b) + F_k
    return StaticForce

path = Path(ROOT_DIR) / "models" / "v2_4" / "urdf" / "dog2_4singleLeg_realconstrast.xml"
experiment_instance = MujocoExperiment(str(path))
fixed_para = {
    'stiffness_MAA': 318.76, # 637.52 / 2,
    'stiffness_BAA': 315.8, # 631.6 / 2,
    'l10': 0.174,
    'l20': 0.2562,
    'damping_MAA': 11.34,
    'damping_BAA': 10.90,
    'c1_thigh': 0,
    'c2_calf': 0,
    's1': 1.0,
    's2': 1.0,
    'P1': 0,        # To be set
    'P2': 0,
    'P1_prime': 0.0,      
    'P2_prime': 0.0       # equal to F when s=1
}

theta1_array = np.linspace(math.pi/6, math.pi*2/3, 11)[1:]
theta2_array = np.linspace(0, math.pi/2, 11)[1:]
exp_para_list = []
exp_data_t_list = []
exp_data_theta1_list = []
exp_data_theta2_list = []

for theta_1_start in tqdm(theta1_array, desc="Theta 1 Start Loop"):
    for theta_2_start in tqdm(theta2_array, desc="Theta 2 Start Loop", leave=False):
        for theta_1_end in tqdm(theta1_array, desc="Theta 1 End Loop", leave=False):
            for theta_2_end in tqdm(theta2_array, desc="Theta 2 End Loop", leave=False):
                fixed_para['P1'] = get_static_func(theta_1_start, theta_2_start)[0]
                fixed_para['P2'] = get_static_func(theta_1_start, theta_2_start)[1]
                fixed_para['P1_prime'] = get_static_func(theta_1_end, theta_2_end)[0]
                fixed_para['P2_prime'] = get_static_func(theta_1_end, theta_2_end)[1]
                time_sim_, theta1_sim_, theta2_sim_, frames_, valid_, valid_last_ = experiment_instance.run(fixed_para, 10, 21, False)
                exp_para_list.append({
                    'theta_1_start': theta_1_start,
                    'theta_2_start': theta_2_start,
                    'theta_1_end': theta_1_end,
                    'theta_2_end': theta_2_end,
                    'F1': fixed_para['P1'],
                    'F2': fixed_para['P2'],
                    'F1_prime': fixed_para['P1_prime'],
                    'F2_prime': fixed_para['P2_prime'],
                    'valid': valid_,
                    'valid_last': valid_last_
                })
                exp_data_t_list.append(time_sim_[0:1000*20])       # About 15s
                exp_data_theta1_list.append(theta1_sim_[0:1000*20])
                exp_data_theta2_list.append(theta2_sim_[0:1000*20])

# Save as csv
exp_para_df = pd.DataFrame(exp_para_list)
exp_para_df.to_csv(DATA_DIR / "swing_exp_para.csv", index=False)

# save npy
np.save(DATA_DIR / "swing_exp_data_t.npy", np.array(exp_data_t_list))
np.save(DATA_DIR / "swing_exp_data_theta1.npy", np.array(exp_data_theta1_list))
np.save(DATA_DIR / "swing_exp_data_theta2.npy", np.array(exp_data_theta2_list))

# copy video for group(2 6 1 7)
fixed_para['P1'] = get_static_func(theta1_array[2], theta2_array[6])[0]
fixed_para['P2'] = get_static_func(theta1_array[2], theta2_array[6])[1]
fixed_para['P1_prime'] = get_static_func(theta1_array[1], theta2_array[7])[0]
fixed_para['P2_prime'] = get_static_func(theta1_array[1], theta2_array[7])[1]
time_sim_, theta1_sim_, theta2_sim_, frames_, valid_, valid_last_ = experiment_instance.run(fixed_para, 10, 21, True)
media.write_video(VIDEO_DIR / "swing_2617.mp4", frames_, fps=60)
