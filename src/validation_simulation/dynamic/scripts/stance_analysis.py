# # Simulated verification of dynamics during STANCE PHASE
#

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io
import pandas as pd

from pathlib import Path
import rootpath
import sys
ROOT_DIR = Path(rootpath.detect())
sys.path.append(str(ROOT_DIR))
from utils.experiment import MujocoExperiment

CURRENT_DIR = Path(__file__).resolve().parent
EXP_DIR = CURRENT_DIR.parent
DATA_DIR = EXP_DIR / 'data'
LOG_DIR = EXP_DIR / 'log'
FIGURE_DIR = EXP_DIR / 'figure'

# Load the data
sim_data = pd.read_csv(DATA_DIR / "stance_exp_data.csv")
t_sim = sim_data['time'].to_numpy()
theta1_sim = sim_data['theta1'].to_numpy()
theta2_sim = sim_data['theta2'].to_numpy()

t_theo = scipy.io.loadmat(DATA_DIR / "matlab_theo" / "t.mat")['t'].flatten()
theta1_theo = scipy.io.loadmat(DATA_DIR / "matlab_theo" / "theta_1.mat")['theta_1'].flatten()
theta2_theo = scipy.io.loadmat(DATA_DIR / "matlab_theo" / "theta_2.mat")['theta_2'].flatten()

# Calculate the error
t_interp = np.linspace(0, 10, 1000)
theta1_sim_interp = np.interp(t_interp, t_sim, theta1_sim)
theta2_sim_interp = np.interp(t_interp, t_sim, theta2_sim)
theta1_theo_interp = np.interp(t_interp, t_theo, theta1_theo)
theta2_theo_interp = np.interp(t_interp, t_theo, theta2_theo)

rmse_theta1 = np.sqrt(np.mean(np.square(theta1_sim_interp-theta1_theo_interp)))
rmse_theta2 = np.sqrt(np.mean(np.square(theta2_sim_interp-theta2_theo_interp)))
mae_theta1 = np.mean(np.abs(theta1_sim_interp-theta1_theo_interp))
mae_theta2 = np.mean(np.abs(theta2_sim_interp-theta2_theo_interp))
maxerror_theta1 = np.max(np.abs(theta1_sim_interp-theta1_theo_interp))
maxerror_theta2 = np.max(np.abs(theta2_sim_interp-theta2_theo_interp))

# Save the error metrics to a CSV file
error_data = pd.DataFrame({
    'rmse_theta1': [rmse_theta1],
    'rmse_theta2': [rmse_theta2],
    'mae_theta1': [mae_theta1],
    'mae_theta2': [mae_theta2],
    'maxerror_theta1': [maxerror_theta1],
    'maxerror_theta2': [maxerror_theta2]
})
error_data.to_csv(LOG_DIR / "stance_error_summary.csv", index=False, float_format='%.6f')

# Plot the step response of the stance phase
from matplotlib import rcParams

rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12+2,                # Default font size (used for axis labels)
    "axes.titlesize": 14+2,          # Title font size
    "axes.labelsize": 12+2,          # Axis label font size
    "xtick.labelsize": 10+2,         # X-axis tick font size
    "ytick.labelsize": 10+2,         # Y-axis tick font size
    "legend.fontsize": 10+2,         # Legend font size
    "figure.titlesize": 14+2,        # Overall figure title font size (if using plt.suptitle)
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
plt.savefig(FIGURE_DIR / 'stance_phase.png', dpi=200, bbox_inches='tight')

plt.show()

plt.figure(figsize=(3, 3))
# --- Top right: Trajectory plot ---
plt.plot(theta1_sim_interp, theta2_sim_interp, zorder=3, linewidth=2, linestyle='-', label='Simulation')  # Line
plt.plot(theta1_theo_interp, theta2_theo_interp, zorder=3, linewidth=2, linestyle='--', label='Theory')  # Line

plt.xlabel(r'$\theta_1$ (rad)')
plt.ylabel(r'$\theta_2$ (rad)')
plt.title(r'Joint Trajectory in $\theta$ Space', fontsize=16)
plt.axis('equal')
# plt.axis([1.1-0.7, 1.1+0.7, -0.1, 1.3])
plt.grid(True)

plt.tight_layout()
plt.legend()

plt.savefig(FIGURE_DIR / 'stance_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()