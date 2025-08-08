# Down stream application: PID tuning
# utilizing MuJoCo Simulation and minimize/differential_evolution optimization
# result analysis version, with python script

import numpy as np
import mediapy as media
from tqdm import tqdm
import sys, datetime, math, mujoco, os
from scipy.stats import qmc
from scipy.optimize import minimize, differential_evolution
import pandas as pd

# Basic Path
import rootpath
from pathlib import Path

ROOT_DIR = rootpath.detect()   # Get the root directory of the project (.git)
CURRENT_DIR = Path(__file__).resolve().parent

sys.path.append(str(Path(ROOT_DIR)))

# plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'figure.titlesize': 20,
})

def analysis_data():
    # load the parameter and experiment data, re-build the exp_log
    exp_log_loaded = []
    EXP_DIR = CURRENT_DIR.parent
    data_array = np.load(EXP_DIR / "data" / "experiment_data.npy")
    parameters_df = pd.read_csv(EXP_DIR / "data" / "parameters.csv")
    for idx, row in parameters_df.iterrows():
        Kp, Ki, Kd, MSE = parameters_df.iloc[idx]
        time_sim = data_array[idx, 0, :]
        theta1_sim = data_array[idx, 1, :]
        theta2_sim = data_array[idx, 2, :]
        u1_sim = data_array[idx, 3, :]
        u2_sim = data_array[idx, 4, :]
        frames_sim = []  # Placeholder for frames
        exp_log_loaded.append({
            'Kp': Kp,
            'Ki': Ki,
            'Kd': Kd,
            'MSE': MSE,
            'time': time_sim,
            'theta1': theta1_sim,
            'theta2': theta2_sim,
            'u1': u1_sim,
            'u2': u2_sim,
            'frames': frames_sim
        })

    def plot_pid_iterations_by_color(exp_log, duration_steady, duration_experiment):
        N = len(exp_log)
        # Generate a truncated and discretized colormap
        base_cmap = matplotlib.colormaps['ocean']  # Can be changed to cividis, viridis, etc.
        colors = base_cmap(np.linspace(0, 0.7, N))  # Truncate first 40%
        truncated_cmap = mcolors.ListedColormap(colors)

        # Use BoundaryNorm to precisely control color block positions
        boundaries = np.arange(N + 1) - 0.5  # [-0.5, 0.5, ..., N-0.5]
        norm = mcolors.BoundaryNorm(boundaries, N)
        
        sm = cm.ScalarMappable(cmap=truncated_cmap, norm=norm)
        sm.set_array([])

        plt.figure(figsize=(8, 5))

        for idx in range(len(exp_log)):
            Kp, Ki, Kd, MSE = exp_log[idx]['Kp'], exp_log[idx]['Ki'], exp_log[idx]['Kd'], exp_log[idx]['MSE']
            time_sim, theta1_sim, theta2_sim, u1_sim, u2_sim, frames_sim = exp_log[idx]['time'], exp_log[idx]['theta1'], exp_log[idx]['theta2'], exp_log[idx]['u1'], exp_log[idx]['u2'], exp_log[idx]['frames']
            color = truncated_cmap(idx)
            plt.plot(time_sim, theta1_sim, linestyle='--', color=color, alpha=0.8, label=None)
            plt.plot(time_sim, theta2_sim, linestyle='-', color=color, alpha=0.8, label=None)

        cbar = plt.colorbar(sm, ax=plt.gca(), ticks=np.arange(N))
        cbar.set_label('Iteration')
        cbar.ax.set_yticklabels([str(i) for i in range(N)])  # Ensure integer labels

        plt.axhline(y=np.pi/2, color='r', linestyle='--', label='Target Theta1')
        plt.axhline(y=0, color='r', linestyle='-', label='Target Theta2')
        plt.xlim(duration_steady - 1, duration_steady + duration_experiment)
        plt.ylim(-np.pi/12, 2/3*np.pi)
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (radians)')
        plt.title('Joint Angles Over Time (Color = PID Iteration)')
        plt.legend(fontsize=8)
        
        custom_lines = [
            Line2D([0], [0], color='g', linestyle='--', label=r'$\theta_1$ trajectory'),
            Line2D([0], [0], color='g', linestyle='-', label=r'$\theta_2$ trajectory'),
            Line2D([0], [0], color='r', linestyle='--', label=r'Target $\theta_1$'),
            Line2D([0], [0], color='r', linestyle='-', label=r'Target $\theta_2$'),
        ]
        plt.legend(handles=custom_lines, 
                    loc='upper right',           # Align legend's upper left corner to this bbox
                    borderaxespad=0.2,           # Distance between legend and main plot border
                    labelspacing=0.2,            # Controls vertical spacing between legend entries (default 0.5, reduced here)
                    ncol=2,                      # Set to two columns, legend will wrap automatically
                )
        plt.grid(True)

        # high resolution
        plt.tight_layout()
        plt.savefig(EXP_DIR / "figure" / "joint_trajectory.png", dpi=300, bbox_inches='tight')  # Save as high-res PNG
        plt.show()

    duration_steady = 10  # seconds, the steady state time
    duration_experiment = 10  # seconds, the experiment time
    plot_pid_iterations_by_color(exp_log_loaded, duration_steady, duration_experiment)

if __name__ == "__main__":
    analysis_data()