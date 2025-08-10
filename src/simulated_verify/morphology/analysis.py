import matplotlib.pyplot as plt

# analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib import rcParams

# file_path
import rootpath
import sys
from pathlib import Path

ROOT_DIR = rootpath.detect()   # Get the root directory of the project (.git)
CURRENT_DIR = Path(__file__).resolve().parent
print(f"ROOT_DIR: {ROOT_DIR}")
print(f"CURRENT_DIR: {CURRENT_DIR}")

sys.path.append(str(Path(ROOT_DIR)))

config = {
    "font.family":'Times New Roman', 
    "axes.unicode_minus": False, 
    "font.size": 15, 
}
rcParams.update(config)

# load and compare
traj_mj = pd.read_csv(CURRENT_DIR / 'test_data/data_mj.csv')
traj_sp = pd.read_csv(CURRENT_DIR / 'test_data/data_sp.csv')

plt.figure(figsize=(8, 5))

colors = plt.get_cmap('tab10')

# Sympy
plt.plot(traj_sp['time'], traj_sp['theta1'], label=r'$\theta_1^{\mathrm{SP}}$', color=colors(0), linestyle='-', linewidth=2, alpha=0.5)
plt.plot(traj_sp['time'], traj_sp['theta2'], label=r'$\theta_2^{\mathrm{SP}}$', color=colors(1), linestyle='-', linewidth=2, alpha=0.5)
plt.plot(traj_sp['time'], traj_sp['theta3'], label=r'$\theta_3^{\mathrm{SP}}$', color=colors(2), linestyle='-', linewidth=2, alpha=0.5)

# MuJoCo
plt.plot(traj_mj['time'], traj_mj['theta1'], label=r'$\theta_1^{\mathrm{MJ}}$', color=colors(0), linestyle='--', linewidth=2, alpha=0.5)
plt.plot(traj_mj['time'], traj_mj['theta2'], label=r'$\theta_2^{\mathrm{MJ}}$', color=colors(1), linestyle='--', linewidth=2, alpha=0.5)
plt.plot(traj_mj['time'], traj_mj['theta3'], label=r'$\theta_3^{\mathrm{MJ}}$', color=colors(2), linestyle='--', linewidth=2, alpha=0.5)

plt.xlabel('Time (s)', fontsize=25)
plt.ylabel('Joint Angle (rad)', fontsize=25)
# plt.title('Joint Angle Comparison between Analytical (Sympy) and EquiMus Simulation (MuJoCo)', fontsize=22)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='right', bbox_to_anchor=(0.98, 0.38), fontsize=18, ncol=2, labelspacing=0.2, borderpad=0.2)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlim(traj_mj['time'].min(), traj_mj['time'].max()/2)
plt.tight_layout()
plt.savefig(CURRENT_DIR / 'figure/joint_angle_comparison.pdf', dpi=500, bbox_inches='tight')
plt.savefig(CURRENT_DIR / 'figure/joint_angle_comparison.jpg', dpi=500, bbox_inches='tight')
plt.show()

# Calculate the RMSE of each theta
print(np.max(traj_sp['theta1'] - traj_mj['theta1']))

# NOTE: must align the time first, then calculate RMSE
# Interpolate traj_sp to traj_mj's time points
theta1_sp_interp = np.interp(traj_mj['time'], traj_sp['time'], traj_sp['theta1'])
theta2_sp_interp = np.interp(traj_mj['time'], traj_sp['time'], traj_sp['theta2'])
theta3_sp_interp = np.interp(traj_mj['time'], traj_sp['time'], traj_sp['theta3'])

# Compute differences
diff_theta1 = traj_mj['theta1'] - theta1_sp_interp
diff_theta2 = traj_mj['theta2'] - theta2_sp_interp
diff_theta3 = traj_mj['theta3'] - theta3_sp_interp

rmse_theta1 = np.sqrt(np.mean(diff_theta1 ** 2))
rmse_theta2 = np.sqrt(np.mean(diff_theta2 ** 2))
rmse_theta3 = np.sqrt(np.mean(diff_theta3 ** 2))

rmse_log = {
    "rmse_theta1": rmse_theta1,
    "rmse_theta2": rmse_theta2,
    "rmse_theta3": rmse_theta3
}
rmse_log = pd.DataFrame([rmse_log])
rmse_log.to_csv(CURRENT_DIR / 'test_data' / 'rmse_log.csv', index=False)
print(rmse_log)