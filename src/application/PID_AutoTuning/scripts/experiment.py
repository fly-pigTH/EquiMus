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
from utils.experiment import MujocoExperiment

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


# In some way, PID experiment is the specific child class of the MujocoExperiment
class Downstream_PID_experiment(object):
    """Downstream PID tuning experiment class.

    This class wraps a MujocoExperiment instance and provides tools to 
    simulate and evaluate PID controllers on a single-leg robotic model.
    """
    def __init__(self):
        path = Path(ROOT_DIR) / "models" / "SingleLeg_ideal.xml"
        # Note: only compatible with string path, not Path object
        self.experiment_instance = MujocoExperiment(str(path), model_type="ideal_geom_swing")
        
        # Model Parameter Set
        params = {
            'stiffness_MAA': 318.76,
            'stiffness_BAA': 315.8,
            'l10': 0.174,
            'l20': 0.2562,
            'damping_MAA': 11.34,
            'damping_BAA': 10.9,
            's1': 0.000654,
            's2': 0.000637,
            'c1_thigh': 0,
            'c2_calf': 0,
            'P1': 0,
            'P2': 0,
            'P1_prime': 0,
            'P2_prime': 0
        }
        self.experiment_instance.apply_params(params)

        # Run the Experiments
        self.framerate = 60

    def simulate_response_evaluate_pid(self, Kp, Ki, Kd, mode="optimize"):
        """Run Mujoco simulation using given PID parameters.

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            mode (str): 'optimize' (returns only MSE) or 'validate' (returns 
                        detailed time series and rendered frames).

        Returns:
           tuple:
            - float: Mean squared error (MSE) of the PID response.
            - np.ndarray or None: Time series (if mode='validate', otherwise None).
            - np.ndarray or None: Theta1 angle over time (if mode='validate', otherwise None).
            - np.ndarray or None: Theta2 angle over time (if mode='validate', otherwise None).
            - np.ndarray or None: Control input for theta1 (if mode='validate', otherwise None).
            - np.ndarray or None: Control input for theta2 (if mode='validate', otherwise None).
            - list[np.ndarray]: Rendered frames (empty list if mode='optimize').

        """

        # exp_set
        duration_steady = 10  # seconds, the steady state time
        duration_response = 10  # seconds, the experiment time (response to the force)

        m, d = self.experiment_instance.model, self.experiment_instance.data

        # Reset state and time
        mujoco.mj_resetData(m, d)

        # Reset state and time
        results = []
        frames = []
        dt = m.opt.timestep

        # PID
        err_sum = [0.0, 0.0]
        last_err = [0.0, 0.0]
        error_list = []
        total_error = 0.0

        with mujoco.Renderer(m, self.experiment_instance.height, self.experiment_instance.width) as renderer:
            while d.time < duration_steady + duration_response:
                theta1 = d.qpos[0] + np.pi / 2
                theta2 = d.qpos[1] + 0
                u1 = 0
                u2 = 0
                l1, l2 = MujocoExperiment.calculate_length(theta1, theta2)
                l1_target, l2_target = MujocoExperiment.calculate_length(math.pi/2, 0)
                if d.time >= duration_steady:
                    err1 = l1_target - l1
                    err2 = l2_target - l2
                    err_sum[0] += err1 * dt
                    err_sum[1] += err2 * dt
                    derr1 = (err1 - last_err[0]) / dt
                    derr2 = (err2 - last_err[1]) / dt
                    u1 = Kp * err1 + Ki * err_sum[0] + Kd * derr1
                    u2 = Kp * err2 + Ki * err_sum[1] + Kd * derr2
                    # limit u1, u2 in [-100, 100]
                    u1 = np.clip(u1, -100, 100)
                    u2 = np.clip(u2, -100, 100)
                    last_err = [err1, err2]
                    d.ctrl[:2] = u1
                    d.ctrl[2:] = u2
                    total_error += err1**2 + err2**2
                    error_list.append(err1**2 + err2**2)
                else:
                    d.ctrl[:] = 0
                # update
                mujoco.mj_step(m, d)
                if mode == "validate":
                    if len(frames) < d.time * self.framerate:   # assume the simulation is running much faster than the rendering
                        renderer.update_scene(d, camera="closeup")
                        frames.append(renderer.render())
                    results.append((d.time, theta1, theta2, u1, u2))
            if mode == "validate":
                time_sim, theta1_sim, theta2_sim, u1, u2 = np.array(results).T
            else: 
                time_sim, theta1_sim, theta2_sim, u1, u2 = None, None, None, None, None
            return np.mean(np.array(error_list)), time_sim, theta1_sim, theta2_sim, u1, u2, frames
        
    # post-process the results: transform the results into csv format / np format
    def save_results(self, exp_log):
        """Save the experiment results to CSV and video files."""
        # save all PID as a csv file
        df = pd.DataFrame({
            'Kp': [expdata['Kp'] for expdata in exp_log],
            'Ki': [expdata['Ki'] for expdata in exp_log],
            'Kd': [expdata['Kd'] for expdata in exp_log],
            'MSE': [expdata['MSE'] for expdata in exp_log]
        })
        data_dir = CURRENT_DIR.parent / "data"
        df.to_csv(data_dir / "parameters.csv", index=False)

        # save time, theta1, theta2, u1, u2, frames as an ndarray
        # squeeze
        data_array = np.stack([
            np.stack([d['time'], d['theta1'], d['theta2'], d['u1'], d['u2']], axis=0)   # index: channel * time, eg, 5*100
            for d in exp_log
        ], axis=0)  # index: experiment * channel * time
        np.save(data_dir / "experiment_data.npy", data_array)

        # save the video
        dt_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        for iteration in tqdm(range(len(exp_log)), desc="Saving video frames"):
            frames = exp_log[iteration]['frames']
            if len(frames) == 0:
                continue
            # save the video
            video_filename = f"PID_{iteration}.mp4"
            video_dir = CURRENT_DIR.parent / "video"
            media.write_video(video_dir / video_filename, frames, fps=self.framerate)

    def make_init_population(self, bounds, prior_sample, popsize=10):
        dim = len(bounds)
        total_num = popsize * dim

        # 1. Latin Hypercube Sampling in [0,1]^dim
        sampler = qmc.LatinHypercube(d=dim, seed=42)    # NOTE: set the seed to control the randomness
        sample_unit = sampler.random(n=total_num - 1)  # -1 to leave space for prior

        # 2. Scale LHS to bounds
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        sample_scaled = qmc.scale(sample_unit, lower_bounds, upper_bounds)

        # 3. Add prior sample (make sure it's within bounds)
        prior_sample_clipped = np.clip(prior_sample, lower_bounds, upper_bounds)
        init_population = np.vstack([sample_scaled, prior_sample_clipped])

        return init_population
    
    def run_optimization(self):
        """Run full PID optimization (L-BFGS-B + Differential Evolution)."""
        print("Begin optimization for PID parameters...")
        exp_log = []

        def pid_objective(x):
            """Objective function for optimization: runs simulation and returns MSE."""
            Kp, Ki, Kd = x
            # Evaluate the PID parameters
            MSE, _time, _theta1, _theta2, _u1, _u2, _frames = self.simulate_response_evaluate_pid(Kp, Ki, Kd, mode="optimize")
            return MSE

        def record_best_x(xk, convergence=None):
            Kp, Ki, Kd = xk
            MSE, time, theta1, theta2, u1, u2, frames = self.simulate_response_evaluate_pid(Kp, Ki, Kd, mode="validate")
            exp_log.append({
                'Kp': Kp,
                'Ki': Ki,
                'Kd': Kd,
                'MSE': MSE,
                'time': time,
                'theta1': theta1,
                'theta2': theta2,
                'u1': u1,
                'u2': u2,
                'frames': frames
            })
            print(f"Evaluating [PID]: Kp={Kp}, Ki={Ki}, Kd={Kd}, Total Error={MSE}")

        # Step1: minimize the PID parameters using L-BFGS-B
        x0 = [100.0, 0.0, 2.0]      # initialization
        bounds = [(1, 10000), (0, 1000), (0, 100)]
        res_local = minimize(pid_objective, x0, bounds=bounds, method='L-BFGS-B', 
                             callback=record_best_x, options={'disp': True, 'maxiter': 1000})
        # NOTE: for the BFGS is a deterministic algorithm, so no need to set seed
        
        # Step2: use the result of L-BFGS-B as the prior sample for differential evolution
        init_population = self.make_init_population(bounds, res_local.x, popsize=10)
        res = differential_evolution(pid_objective, init=init_population, bounds=bounds, 
                                     popsize=10, disp=True, callback=record_best_x,  
                                     strategy='rand1bin', mutation=(0.05, 0.2), seed=42)
        print("Best PID:", res.x)
        return exp_log

def generate_PID_trajectories():
    mySim = Downstream_PID_experiment()
    exp_log = mySim.run_optimization()
    mySim.save_results(exp_log)

if __name__ == "__main__":
    # step 1: generate the control data and calculate the MSE of the joint angle
    generate_PID_trajectories()