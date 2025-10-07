# Simplified and complete experimental version derived from theo.ipynb
# Refactored using utils-exp_func

import numpy as np
import math
import mediapy as media
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import rootpath
import sys
ROOT_DIR = rootpath.detect()
sys.path.append(str(Path(ROOT_DIR)))
from utils.experiment import MujocoExperiment

from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
EXP_DIR = CURRENT_DIR.parent


# theta->static force (theoretical)
def get_static_func(theta_1, theta_2, k_3=637.52/2, k_4=631.6/2, l_10=0.174, l_20=0.2562, m_3 = 0.18648, m_4=0.27266):
    """
    Calculate static forces based on given parameters.
    - theta_1: Angle of joint 1 (in radians)
    - theta_2: Angle of joint 2 (in radians)
    - k_3: Stiffness of spring 1
    - k_4: Stiffness of spring 2
    - l_10: Rest length of spring 1
    - l_20: Rest length of spring 2
    - m_3: Mass at joint 1
    - m_4: Mass at joint 2
    """

    a_1 = 0.25
    a_2 = 0.25
    b_1 = 0.21213
    b_2 = 0.1
    d_1 = 0.06
    d_2 = 0.10
    # s = 0

    # M = 0.155
    m_1, m_2 = 0.086, 0.1033
    # I_1, I_2 = (1/12 * m_1 * 0.25**2), (1/12 * m_2 * 0.25**2)

    # m_3, m_4 = 0.18648, 0.27266

    # k_3, k_4 = 637.52/2, 631.6/2

    # c_3, c_4 = 22.68/2, 21.8/2

    g = 9.8

    beta_1 = 8.13 / 180 * np.pi
    beta_2 = 30 / 180 * np.pi

    # l_10 = 0.174
    # l_20 = 0.252

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


    l_1 = np.sqrt((A_x_O - C_x_O)**2 + (A_y_O - C_y_O)**2)
    l_2 = np.sqrt((B_x_O - D_x_O)**2 + (B_y_O - D_y_O)**2)

    dl_1_dtheta_1 = 1/l_1 * ((A_x_O - C_x_O) * (dA_x_O_dtheta_1 - dC_x_O_dtheta_1) + (A_y_O - C_y_O) * (dA_y_O_dtheta_1 - dC_y_O_dtheta_1))
    dl_2_dtheta_1 = 1/l_2 * ((B_x_O - D_x_O) * (dB_x_O_dtheta_1 - dD_x_O_dtheta_1) + (B_y_O - D_y_O) * (dB_y_O_dtheta_1 - dD_y_O_dtheta_1))

    dl_1_dtheta_2 = 1/l_1 * ((A_x_O - C_x_O) * (dA_x_O_dtheta_2 - dC_x_O_dtheta_2) + (A_y_O - C_y_O) * (dA_y_O_dtheta_2 - dC_y_O_dtheta_2))
    dl_2_dtheta_2 = 1/l_2 * ((B_x_O - D_x_O) * (dB_x_O_dtheta_2 - dD_x_O_dtheta_2) + (B_y_O - D_y_O) * (dB_y_O_dtheta_2 - dD_y_O_dtheta_2))

    RHSb_1 = -( m_1*g*(dE_y_O_dtheta_1/2) + m_2*g*((dE_y_O_dtheta_1+dF_y_O_dtheta_1)/2) + m_3*g*(dC_y_O_dtheta_1/2) + m_4*g*(dD_y_O_dtheta_1/2) )
    RHSb_2 = -( m_1*g*(dE_y_O_dtheta_2/2) + m_2*g*((dE_y_O_dtheta_2+dF_y_O_dtheta_2)/2) + m_3*g*(dC_y_O_dtheta_2/2) + m_4*g*(dD_y_O_dtheta_2/2) )
    b = np.array([RHSb_1, RHSb_2])

    LHSA = np.array([[dl_1_dtheta_1, dl_2_dtheta_1], 
                    [dl_1_dtheta_2, dl_2_dtheta_2]])

    F_k = np.array([k_3*(l_1-l_10), k_4*(l_2-l_20)])

    StaticForce = np.linalg.solve(LHSA, b) + F_k
    return StaticForce

# ## Video Recording - A case of simulated verification

theta_1_tar_array = np.linspace(math.pi/6, math.pi*2/3, 11)[1:]
theta_2_tar_array = np.linspace(0, math.pi/2, 11)[1:]
path = (ROOT_DIR + "/models/v2_4/urdf/dog2_4singleLeg_realconstrast.xml")

def make_demo():
    ''' A demo for video making, show the static state of the robotic leg '''
    np.random.seed(42)
    exp = MujocoExperiment(path)
    para_init = {
        'stiffness_MAA': 318.76, #637.52 / 2,
        'stiffness_BAA': 315.8, #631.6 / 2,
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

    para_record_list = []
    exp_data = []
    frames_last_group = None

    # set parameter
    para = para_init.copy()     # create new parameter
    para['stiffness_MAA'] *= np.random.uniform(0.5, 1.5)
    para['stiffness_BAA'] *= np.random.uniform(0.5, 1.5)
    para['l10'] *= np.random.uniform(0.5, 1.5)
    para['l20'] *= np.random.uniform(0.5, 1.5)
    para_record_list.append(para)
    # print(f"Para: {para}")

    for i in tqdm(range(len(theta_1_tar_array))):
        for j in range(len(theta_2_tar_array)):
            theta_1_theo = theta_1_tar_array[i]
            theta_2_theo = theta_2_tar_array[j]
            force_theoretical = get_static_func(theta_1_theo, theta_2_theo, para['stiffness_MAA'], para['stiffness_BAA'], para['l10'], para['l20'])
            para['P1_prime'] = force_theoretical[0]
            para['P2_prime'] = force_theoretical[1]
            time_sim_, theta1_sim_, theta2_sim_, frames_, valid_, valid_last_ = exp.run(para, time_step=0, duration=10, ifrender=True)
            # Save Video
            media.write_video(EXP_DIR / "video" / f"experiment_output_{i*10+j}.mp4", frames_, fps=60)

            exp_data.append({
                'valid': valid_,
                'valid_last': valid_last_,
                'theta_1_theo': theta_1_theo,
                'theta_2_theo': theta_2_theo,
                'theta1_sim_': theta1_sim_[-1],
                'theta2_sim_': theta2_sim_[-1],
                'F1_apply': force_theoretical[0],
                'F2_apply': force_theoretical[1],
            })

    exp_data = pd.DataFrame(data=exp_data)
    exp_data.to_csv(EXP_DIR / "data" / "demo" / "experiment_data.csv", index=False)
    para_record_list = pd.DataFrame(data=para_record_list)
    para_record_list.to_csv(EXP_DIR / "data" / "demo" / "experiment_para.csv", index=False)

# ## Static Response of Multi Group of parameteres

def run_one_experiment(exp, para):
    '''one group of parameter, scan 10*10 Force Commands'''
    exp_data = []

    for i in tqdm(range(len(theta_1_tar_array))):
        for j in range(len(theta_2_tar_array)):
            theta_1_theo = theta_1_tar_array[i]
            theta_2_theo = theta_2_tar_array[j]
            force_theoretical = get_static_func(theta_1_theo, theta_2_theo, para['stiffness_MAA'], para['stiffness_BAA'], para['l10'], para['l20'])
            para['P1_prime'] = force_theoretical[0]
            para['P2_prime'] = force_theoretical[1]
            time_sim_, theta1_sim_, theta2_sim_, frames_, valid_, valid_last_ = exp.run(para, time_step=0, duration=10, ifrender=False)
            # Save Video
            # media.write_video(f'./video/experiment_output_{i*10+j}.mp4', frames_, fps=60)

            if valid_last_:
                # record the data
                exp_data.append({
                    'theta_1_theo': theta_1_theo,
                    'theta_2_theo': theta_2_theo,
                    'theta_1_sim': theta1_sim_[-1],
                    'theta_2_sim': theta2_sim_[-1],
                    'F1_apply': force_theoretical[0],
                    'F2_apply': force_theoretical[1],
                })
    exp_data = pd.DataFrame(data=exp_data)
    # cal error from valid data
    theta_1_theo_array = exp_data['theta_1_theo'].to_numpy()
    theta_2_theo_array = exp_data['theta_2_theo'].to_numpy()
    theta_1_sim_array = exp_data['theta_1_sim'].to_numpy()
    theta_2_sim_array = exp_data['theta_2_sim'].to_numpy()
    # Calculate RMSE, MAE, MaxAE
    RMSE_theta_1 = np.sqrt(np.mean((theta_1_theo_array - theta_1_sim_array)**2))
    RMSE_theta_2 = np.sqrt(np.mean((theta_2_theo_array - theta_2_sim_array)**2))
    MAE_theta_1 = np.mean(np.abs(theta_1_theo_array - theta_1_sim_array))
    MAE_theta_2 = np.mean(np.abs(theta_2_theo_array - theta_2_sim_array))
    MaxAE_theta_1 = np.max(np.abs(theta_1_theo_array - theta_1_sim_array))
    MaxAE_theta_2 = np.max(np.abs(theta_2_theo_array - theta_2_sim_array))
    return {
        'k3': para['stiffness_MAA'],
        'k4': para['stiffness_BAA'],
        'l10': para['l10'],
        'l20': para['l20'],
        'RMSE_theta_1': RMSE_theta_1,
        'RMSE_theta_2': RMSE_theta_2,
        'MAE_theta_1': MAE_theta_1,
        'MAE_theta_2': MAE_theta_2,
        'MaxAE_theta_1': MaxAE_theta_1,
        'MaxAE_theta_2': MaxAE_theta_2,
        'valid_exp_num': len(exp_data)
    }

def run_batch_experiments():
    exp = MujocoExperiment(path)
    para_init = {
        'stiffness_MAA': 318.76, #637.52 / 2,
        'stiffness_BAA': 315.8, #631.6 / 2,
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
    multi_exp_data = []

    for i in range(100):
        # randomly set stiffness and l10, l20
        para = para_init.copy()     # create new parameter
        para['stiffness_MAA'] *= np.random.uniform(0.5, 1.5)
        para['stiffness_BAA'] *= np.random.uniform(0.5, 1.5)
        para['l10'] *= np.random.uniform(0.5, 1.5)
        para['l20'] *= np.random.uniform(0.5, 1.5)

        # run one experiment
        single_experiment_res_dic = run_one_experiment(exp, para)
        if single_experiment_res_dic['valid_exp_num'] == 0:
            print("No valid experiment data.")
            continue
        multi_exp_data.append(single_experiment_res_dic)
    return multi_exp_data

def run_whole_exp():
    np.random.seed(42)
    multi_exp_data = run_batch_experiments()
    multi_exp_data = pd.DataFrame(data=multi_exp_data)
    multi_exp_data.to_csv(EXP_DIR / "data" / "multi_exps_data.csv", index=False)

def data_analysis():
    # Load the data
    multi_exp_data = pd.read_csv(EXP_DIR / "data" / "multi_exps_data.csv")

    # Calculate RMSE, MAE, MaxAE
    RMSE_theta_1 = np.sqrt(np.mean((multi_exp_data['RMSE_theta_1'])**2))
    RMSE_theta_2 = np.sqrt(np.mean((multi_exp_data['RMSE_theta_2'])**2))
    MAE_theta_1 = np.mean(np.abs(multi_exp_data['MAE_theta_1']))
    MAE_theta_2 = np.mean(np.abs(multi_exp_data['MAE_theta_2']))
    MaxAE_theta_1 = np.max(np.abs(multi_exp_data['MaxAE_theta_1']))
    MaxAE_theta_2 = np.max(np.abs(multi_exp_data['MaxAE_theta_2']))

    # save as csv
    overall_error = [{
        'RMSE_theta_1': RMSE_theta_1,
        'RMSE_theta_2': RMSE_theta_2,
        'MAE_theta_1': MAE_theta_1,
        'MAE_theta_2': MAE_theta_2,
        'MaxAE_theta_1': MaxAE_theta_1,
        'MaxAE_theta_2': MaxAE_theta_2
    }]
    overall_error = pd.DataFrame(data=overall_error)
    overall_error.to_csv(EXP_DIR / "data" / "multi_exps_overall_error.csv", index=False)

def exp():
    make_demo()
    run_whole_exp()
    data_analysis()

if __name__ == "__main__":
    exp()