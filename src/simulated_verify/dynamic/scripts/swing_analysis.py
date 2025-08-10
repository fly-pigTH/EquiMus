# # Simulated verificaition of dynamics on SWING PHASE

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import ipywidgets as widgets
import rootpath
import numpy as np
from scipy.io import loadmat
from matplotlib import rcParams

from pathlib import Path
import rootpath
import sys
ROOT_DIR = Path(rootpath.detect())
sys.path.append(str(ROOT_DIR))
from utils.experiment import MujocoExperiment

def analysis_and_plot_swing():
  CURRENT_DIR = Path(__file__).resolve().parent
  EXP_DIR = CURRENT_DIR.parent
  DATA_DIR = EXP_DIR / 'data'
  LOG_DIR = EXP_DIR / 'log'
  FIGURE_DIR = EXP_DIR / 'figure'

  # Data from theotical sim (Matlab)
  t_bias = 10     # time shift for the sim exp start from 10s
  t_sim_list = np.load(DATA_DIR / 'swing_exp_data_t.npy', allow_pickle=True) - t_bias
  theta1_sim_list = np.load(DATA_DIR / 'swing_exp_data_theta1.npy', allow_pickle=True)
  theta2_sim_list = np.load(DATA_DIR / 'swing_exp_data_theta2.npy', allow_pickle=True)

  # Data from equivalent sim (Matlab)

  data = loadmat(DATA_DIR / 'matlab_theo' / 'swing_expdata.mat', simplify_cells=True)
  t_all = data['t_all']       # 10000 * n (list of array)
  Xt_all = data['Xt_all']     # 10000 * n * 4 (list of 2d array)

  # Test One Group
  exp_id = 11

  # step_response_compare Function; point data for indexing
  def step_response_compare(t_array_matlab, theta1_array_matlab, theta2_array_matlab, t_array_mujoco, theta1_array_mujoco, theta2_array_mujoco):
    """
    Compare the step response between theoretical (Matlab) and simulated (Mujoco) data.

    Parameters:
    - t_array_matlab (numpy.ndarray): Time array for Matlab data.
    - theta1_array_matlab (numpy.ndarray): Theta1 values from Matlab data.
    - theta2_array_matlab (numpy.ndarray): Theta2 values from Matlab data.
    - t_array_mujoco (numpy.ndarray): Time array for Mujoco data.
    - theta1_array_mujoco (numpy.ndarray): Theta1 values from Mujoco data.
    - theta2_array_mujoco (numpy.ndarray): Theta2 values from Mujoco data.

    Returns:
    - dict: A dictionary containing:
      - 'valid' (bool): Whether the data is within valid ranges.
      - 'RMSE_theta1' (float): Root Mean Square Error for Theta1.
      - 'RMSE_theta2' (float): Root Mean Square Error for Theta2.
      - 'MAE_theta1' (float): Mean Absolute Error for Theta1.
      - 'MAE_theta2' (float): Mean Absolute Error for Theta2.
      - 'MaxAE_theta1' (float): Maximum Absolute Error for Theta1.
      - 'MaxAE_theta2' (float): Maximum Absolute Error for Theta2.
    """
    # Calculate the MSE using interpolation over the interval [0-10s]
    t_interp = np.linspace(0, 10, 1000)
    theta1_sim_interp = np.interp(t_interp, t_array_mujoco, theta1_array_mujoco)
    theta2_sim_interp = np.interp(t_interp, t_array_mujoco, theta2_array_mujoco)
    theta1_theo_interp = np.interp(t_interp, t_array_matlab, theta1_array_matlab)
    theta2_theo_interp = np.interp(t_interp, t_array_matlab, theta2_array_matlab)

    # Check validity of data
    valid = True
    min_theta1 = min(np.min(theta1_sim_interp), np.min(theta1_theo_interp))
    max_theta1 = max(np.max(theta1_sim_interp), np.max(theta1_theo_interp))
    min_theta2 = min(np.min(theta2_sim_interp), np.min(theta2_theo_interp))
    max_theta2 = max(np.max(theta2_sim_interp), np.max(theta2_theo_interp))
    threshold = math.pi / 180  # 1 degree in radians
    if ((np.pi / 6 + threshold) >= min_theta1 or 
      max_theta1 >= (np.pi / 6 + 2 / 3 * np.pi - threshold) or 
      (0 + threshold) >= min_theta2 or 
      max_theta2 >= (0 + 2 / 3 * np.pi - threshold)):
      valid = False

    # Calculate errors
    RMSE_theta1 = np.sqrt(np.mean(np.square(theta1_sim_interp - theta1_theo_interp)))
    RMSE_theta2 = np.sqrt(np.mean(np.square(theta2_sim_interp - theta2_theo_interp)))
    MAE_theta1 = np.mean(np.abs(theta1_sim_interp - theta1_theo_interp))
    MAE_theta2 = np.mean(np.abs(theta2_sim_interp - theta2_theo_interp))
    MaxAE_theta1 = np.max(np.abs(theta1_sim_interp - theta1_theo_interp))
    MaxAE_theta2 = np.max(np.abs(theta2_sim_interp - theta2_theo_interp))

    return {
      'valid': valid,
      'RMSE_theta1': RMSE_theta1,
      'RMSE_theta2': RMSE_theta2,
      'MAE_theta1': MAE_theta1,
      'MAE_theta2': MAE_theta2,
      'MaxAE_theta1': MaxAE_theta1,
      'MaxAE_theta2': MaxAE_theta2
    }

  # Calculate the error for all data, save in csv file
  swing_error_list = []
  for i in tqdm(range(len(t_all))):
    result = step_response_compare(
      t_all[i], 
      Xt_all[i][:, 0], 
      Xt_all[i][:, 1], 
      t_sim_list[i], 
      theta1_sim_list[i], 
      theta2_sim_list[i]
    )
    swing_error_list.append(result)
  # save as pd
  swing_error_list = pd.DataFrame(swing_error_list)
  swing_error_list.to_csv(DATA_DIR / 'swing_error_list.csv', index=False, float_format='%.6f')

  # Load saved data and analysis
  swing_error_list = pd.read_csv(DATA_DIR / 'swing_error_list.csv')
  swing_error_valid = swing_error_list.query('valid==True')

  # Summary
  max_RMSE_theta1 = swing_error_valid['RMSE_theta1'].max()
  max_RMSE_theta2 = swing_error_valid['RMSE_theta2'].max()
  max_MAE_theta1 = swing_error_valid['MAE_theta1'].max()
  max_MAE_theta2 = swing_error_valid['MAE_theta2'].max()
  max_MaxAE_theta1 = swing_error_valid['MaxAE_theta1'].max()
  max_MaxAE_theta2 = swing_error_valid['MaxAE_theta2'].max()
  # save to error_summary file
  error_summary = {
    'max_RMSE_theta1': max_RMSE_theta1,
    'max_RMSE_theta2': max_RMSE_theta2,
    'max_MAE_theta1': max_MAE_theta1,
    'max_MAE_theta2': max_MAE_theta2,
    'max_MaxAE_theta1': max_MaxAE_theta1,
    'max_MaxAE_theta2': max_MaxAE_theta2
  }
  error_summary = pd.DataFrame(error_summary, index=[0])
  error_summary.to_csv(LOG_DIR / 'swing_error_summary.csv', index=False, float_format='%.6f')

  # Show single experiment data

  # Query the theta
  Theta1_array = np.linspace(math.pi/6, math.pi*2/3, 11)[1:]
  Theta2_array = np.linspace(0, math.pi/2, 11)[1:]

  # Slide creation
  slider_theta1_index_start = widgets.IntSlider(value=0, min=0, max=len(Theta1_array)-1, step=1, description=r'$\theta_1$ Start Index')
  slider_theta1_index_end = widgets.IntSlider(value=0, min=0, max=len(Theta1_array)-1, step=1, description=r'$\theta_1$ End Index')
  slider_theta2_index_start = widgets.IntSlider(value=0, min=0, max=len(Theta2_array)-1, step=1, description=r'$\theta_2$ Start Index')
  slider_theta2_index_end = widgets.IntSlider(value=0, min=0, max=len(Theta2_array)-1, step=1, description=r'$\theta_2$ End Index')

  valid_id = 2167
  t_interp = np.linspace(0, 10, 1000)
  theta1_sim_interp = np.interp(t_interp, t_sim_list[exp_id], theta1_sim_list[exp_id])
  theta2_sim_interp = np.interp(t_interp, t_sim_list[exp_id], theta2_sim_list[exp_id])
  theta1_theo_interp = np.interp(t_interp, t_all[exp_id], Xt_all[exp_id][:, 0])
  theta2_theo_interp = np.interp(t_interp, t_all[exp_id], Xt_all[exp_id][:, 1]) 

  # Draw the step response of Stance Phase
  from matplotlib import rcParams

  rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,                # Default font size (used for axis labels)
    "axes.titlesize": 16,          # Title font size
    "axes.labelsize": 14,          # Axis label font size
    "xtick.labelsize": 12,         # X-axis tick font size
    "ytick.labelsize": 12,         # Y-axis tick font size
    "legend.fontsize": 12,         # Legend font size
    "figure.titlesize": 16,        # Overall figure title font size (if using plt.suptitle)
  })


  plt.figure(figsize=(6, 3))

  plt.plot(t_interp, theta1_sim_interp, label=r'$\theta_1$ (sim)', color='blue', linewidth=2, linestyle='-')

  plt.plot(t_interp, theta2_sim_interp, label=r'$\theta_2$ (sim)', color='green', linewidth=2, linestyle='-')

  plt.plot(t_interp, theta1_theo_interp, label=r'$\theta_1$ (theory)', color='orange', linewidth=2, linestyle='--')

  plt.plot(t_interp, theta2_theo_interp, label=r'$\theta_2$ (theory)', color='red', linewidth=2, linestyle='--')

  # Titles and labels
  plt.title("Comparison of Simulated and Theoretical Joint Angles", fontsize=16)
  plt.xlabel("Time (s)")
  plt.ylabel("Joint Angle (rad)")

  # Legend, grid, layout
  plt.legend(ncol=2, loc="best")
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(FIGURE_DIR / 'swing_phase2167.png', dpi=200, bbox_inches='tight')
  plt.show()

  plt.figure(figsize=(3, 3))
  # --- Top right: Trajectory plot ---
  plt.plot(theta1_sim_interp, theta2_sim_interp, zorder=3, linewidth=2, linestyle='-', label='Simulation')  # Line
  plt.plot(theta1_theo_interp, theta2_theo_interp, zorder=3, linewidth=2, linestyle='--', label='Theory')  # Line

  plt.xlabel(r'$\theta_1$ (rad)')
  plt.ylabel(r'$\theta_2$ (rad)')
  plt.title(r'Joint Trajectory in $\theta$ Space', fontsize=16)
  plt.grid()
  plt.axis('equal')
  plt.tight_layout()
  plt.legend()
  # plt.gca().set_aspect('equal', adjustable='box')
  plt.savefig(FIGURE_DIR / 'swing_trajectory.png', dpi=300, bbox_inches='tight')
  plt.show()

if __name__ == "__main__":
    analysis_and_plot_swing()