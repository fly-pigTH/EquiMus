# General experimental results for testing system response and providing video recording results
# As generic as possible, serving as a prototype for future reinforcement learning

# Topology randomization

import mujoco
import numpy as np
import math
import matplotlib.pyplot as plt
import mediapy as media
from tqdm import tqdm
import time

'''
    Using the same model file, parameters can be modified
    Input: model parameters, experiment time, step time, F1, F2, F1', F2'
    Output: trajectory of state, rendering pixels array
'''

class MujocoExperiment(object):
    def __init__(self, model_path, model_type="real_geom"):
        self.model_path = model_path
        self.model = None   # wait for instantiation
        self.data = None
        self.height = 320
        self.width = 400
        self.framerate = 60  # (Hz)
        self.model_type = model_type        # default: swing

    @staticmethod
    def calculate_length(theta1, theta2):
        a1, a2 = 0.25, 0.25
        b1, b2 = 0.21213, 0.1
        d1, d2 = 0.06, 0.10
        beta1, beta2 = np.radians(8.13), np.radians(30)

        C_x, C_y = b1 * np.cos(theta1 - beta1), b1 * np.sin(theta1 - beta1)
        D_x = a1 * np.cos(theta1) + b2 * np.cos(theta1 + theta2 + beta2)
        D_y = a1 * np.sin(theta1) + b2 * np.sin(theta1 + theta2 + beta2)

        l1 = np.hypot(d1 - C_x, 0-C_y)
        l2 = np.hypot(-d2 - D_x, 0-D_y)

        return l1, l2

    def calculate_bias(self, l10, l20):
        if self.model_type == "real_geom":
            l1, l2 = MujocoExperiment.calculate_length(np.pi/2, np.pi/2) # NOTE: the initial model state during loading is theta1=90, theta2=90
        else:   # ideal_geom
            l1, l2 = MujocoExperiment.calculate_length(np.pi/2, 0)
        l1_rel = l10 - l1
        l2_rel = l20 - l2
        return l1_rel / 2, l2_rel / 2

    def apply_params(self, params):
        # Update model for the experiment
        _model = mujoco.MjModel.from_xml_path(self.model_path)
        _data = mujoco.MjData(_model)
        
        # Set joint damping and stiffness
        joint_ids = ["MAA", "MAA_FM", "BAA", "BAA_FM"]
        for joint in joint_ids:
            idx = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, f"RB_{joint}_SlideJoint")
            assert idx != -1, f"Joint {joint} not found in model"
            stiffness = params['stiffness_BAA' if 'BAA' in joint else 'stiffness_MAA']
            damping = params['damping_BAA' if 'BAA' in joint else 'damping_MAA']
            _model.jnt_stiffness[idx] = stiffness * 2
            _model.dof_damping[idx] = damping * 2

        # Set shoulder and elbow damping
        shoulder_idx = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "RB_Shoulder")
        elbow_idx = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "RB_Elbow")
        assert shoulder_idx != -1, "Shoulder joint not found in model"
        assert elbow_idx != -1, "Elbow joint not found in model"
        _model.dof_damping[shoulder_idx] = params['c1_thigh']
        _model.dof_damping[elbow_idx] = params['c2_calf']

        # Set bias
        l1_bias, l2_bias = self.calculate_bias(params['l10'], params['l20'])
        print(f"l1_bias: {l1_bias}, l2_bias: {l2_bias}")
        for joint in joint_ids:
            idx = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, f"RB_{joint}_SlideJoint")
            assert idx != -1, f"Joint {joint} not found in model"
            _model.qpos_spring[idx] = l1_bias if 'MAA' in joint else l2_bias

        self.model = _model
        self.data = _data
    
    @staticmethod
    def F1_func(t): return 0 if t < 10 else ((t//10)%3 + 1) * 2
    
    @staticmethod
    def F2_func(t): return 0 if t < 10 else (((t//10)+1)%3 + 1) * 2
    
    @staticmethod
    def F3_func(t): return 0 if t < 10 else (((t//10)+2)%3 + 1) * 2
    
    def run(self, params, time_step, duration, ifrender=True):
        self.apply_params(params)
        m, d = self.model, self.data
        # Reset state and time
        mujoco.mj_resetData(m, d)

        results = []
        frames = []
        valid = True        # Check if the angle exceeds the limit
        valid_last = True

        with mujoco.Renderer(m, self.height, self.width) as renderer:
            while d.time < duration:
                # d.ctrl[:2] = params['P1'] * params['s1'] if d.time < time_step else params['P1_prime'] * params['s1']
                # d.ctrl[2:4] = params['P2'] * params['s2'] if d.time < time_step else params['P2_prime'] * params['s2']
                # d.ctrl[4:] = params['P3'] * params['s3'] if d.time < time_step else params['P3_prime'] * params['s3']

                d.ctrl[:2] = MujocoExperiment.F1_func(d.time)
                d.ctrl[2:4] = MujocoExperiment.F2_func(d.time)
                d.ctrl[4:] = MujocoExperiment.F3_func(d.time)
                
                if self.model_type == "real_geom":
                    theta1 = d.qpos[0] + np.pi / 2
                    theta2 = d.qpos[1] + np.pi / 2
                elif self.model_type == "ideal_geom_swing":     
                    theta1 = d.qpos[0] + np.pi / 2
                    theta2 = d.qpos[1] + 0
                    theta3 = d.qpos[2] + 0
                else:   # ideal geom    # ideal_geom_stance, with 2 DOF
                    theta1 = d.qpos[2] + np.pi / 2
                    theta2 = d.qpos[3] + 0
                    
                results.append((d.time, theta1, theta2, theta3))
                # check NOTE: only consider the real geom
                # TODO: add threshold to judge the <=
                threshold = math.pi/180     # 1 deg in rad
                # NOTE: Here we wet the range of angle to be [0, pi*2/3]
                if ((np.pi/6+threshold) >= (theta1)) or ((theta1) >= (np.pi/6 + 2/3*np.pi-threshold)) or ((0+threshold) >= (theta2)) or ((theta2) >= (0+2/3*np.pi-threshold)):
                    valid = False

                mujoco.mj_step(m, d)

                if len(frames) < d.time * self.framerate:   # assume the simulation is running much faster than the rendering
                    if ifrender:
                        renderer.update_scene(d, camera="closeup")
                        frames.append(renderer.render())
            if ((np.pi/6+threshold) >= (theta1)) or ((theta1) >= (np.pi/6 + 2/3*np.pi-threshold)) or ((0+threshold) >= (theta2)) or ((theta2) >= (0+2/3*np.pi-threshold)):
                valid_last = False

        time_sim, theta1_sim, theta2_sim, theta3_sim = np.array(results).T
        return time_sim, theta1_sim, theta2_sim, theta3_sim, frames, valid, valid_last
    
    def run_with_valve_dynamics(self, params, time_step, duration, ifrender=True):

        self.apply_params(params)
        m, d = self.model, self.data
        # Reset state and time
        mujoco.mj_resetData(m, d)

        # === Discretized proportional valve model parameters ===
        T = m.opt.timestep               # Discrete sampling period
        tau = params.get('tau', 0.2171)  # Time constant τ (unit: seconds)
        a = np.exp(-T / tau)             # Discrete system coefficient
        b = 1 - a

        # Initial valve control output (internal state of the controller)
        valve1 = 0.0
        valve2 = 0.0

        results = []
        pressure_states = []  # To store pressure states
        frames = []
        valid = True        # Check if the angle exceeds the limit
        valid_last = True

        with mujoco.Renderer(m, self.height, self.width) as renderer:
            while d.time < duration:
            # === Input signal (step) ===
                u1 = params['P1'] if d.time < time_step else params['P1_prime']
                u2 = params['P2'] if d.time < time_step else params['P2_prime']

                # === Valve dynamic response (discrete first-order system) ===
                valve1 = a * valve1 + b * u1
                valve2 = a * valve2 + b * u2

                # === Set control signals ===
                d.ctrl[:2] = valve1 * params['s1']
                d.ctrl[2:] = valve2 * params['s2']

                if self.model_type == "real_geom":
                    theta1 = d.qpos[0] + np.pi / 2
                    theta2 = d.qpos[1] + np.pi / 2
                elif self.model_type == "ideal_geom_swing":     
                    theta1 = d.qpos[0] + np.pi / 2
                    theta2 = d.qpos[1] + 0
                else:   # ideal geom    # ideal_geom_stance, with 2 DOF
                    theta1 = d.qpos[2] + np.pi / 2
                    theta2 = d.qpos[3] + 0
                results.append((d.time, theta1, theta2))
                pressure_states.append((valve1, valve2))  # Record current valve state
                # Check NOTE: only consider the real geometry
                # TODO: Add threshold to judge the <=
                threshold = math.pi / 180  # 1 degree in radians
                # NOTE: Here we set the range of angle to be [0, pi*2/3]
                if ((np.pi / 6 + threshold) >= (theta1)) or ((theta1) >= (np.pi / 6 + 2 / 3 * np.pi - threshold)) or ((0 + threshold) >= (theta2)) or ((theta2) >= (0 + 2 / 3 * np.pi - threshold)):
                    valid = False

                mujoco.mj_step(m, d)

                if len(frames) < d.time * self.framerate:   # Assume the simulation is running much faster than the rendering
                    if ifrender:
                        renderer.update_scene(d, camera="closeup")
                        frames.append(renderer.render())
                if ((np.pi / 6 + threshold) >= (theta1)) or ((theta1) >= (np.pi / 6 + 2 / 3 * np.pi - threshold)) or ((0 + threshold) >= (theta2)) or ((theta2) >= (0 + 2 / 3 * np.pi - threshold)):
                    valid_last = False
        time_sim, theta1_sim, theta2_sim = np.array(results).T
        pressure1_sim, pressure2_sim = np.array(pressure_states).T
        return time_sim, theta1_sim, theta2_sim, frames, valid, valid_last, pressure1_sim, pressure2_sim
    
    def fastrun(self, params, time_step, duration, ifrender=True):
        self.apply_params(params)
        m, d = self.model, self.data
        # Reset state and time
        mujoco.mj_resetData(m, d)

        results = []
        frames = []
        valid = True        # Check if the angle exceeds the limit
        valid_last = True
        threshold = math.pi/180     # 1 degree in radians
        theta1_min, theta1_max = np.pi/6 + threshold, np.pi/6 + 2/3*np.pi - threshold
        theta2_min, theta2_max = 0 + threshold, 0 + 2/3*np.pi - threshold

        # Estimate the maximum number of steps
        max_steps = int(duration / 0.001) + 1
        results = np.zeros((max_steps, 3))  # (time, theta1, theta2)
        i = 0

        while d.time < duration:
            d.ctrl[:2] = params['P1'] * params['s1'] if d.time < time_step else params['P1_prime'] * params['s1']
            d.ctrl[2:] = params['P2'] * params['s2'] if d.time < time_step else params['P2_prime'] * params['s2']
            
            if self.model_type == "real_geom":
                theta1 = d.qpos[0] + np.pi / 2
                theta2 = d.qpos[1] + np.pi / 2
            elif self.model_type == "ideal_geom_swing":     
                theta1 = d.qpos[0] + np.pi / 2
                theta2 = d.qpos[1] + 0
            else:   # ideal geometry (ideal_geom_stance, with 2 DOF)
                theta1 = d.qpos[2] + np.pi / 2
                theta2 = d.qpos[3] + 0
            # Store results
            results[i] = (d.time, theta1, theta2)
            i += 1
            # Check if the angles are within the valid range
            if not (theta1_min <= theta1 <= theta1_max and theta2_min <= theta2 <= theta2_max):
                valid = False
            mujoco.mj_step(m, d)
        # Final validation check
        if not (theta1_min <= theta1 <= theta1_max and theta2_min <= theta2 <= theta2_max):
            valid_last = False
        frames = None
        results = results[:i]  # Trim to the actual valid portion
        time_sim, theta1_sim, theta2_sim = np.array(results).T
        return time_sim, theta1_sim, theta2_sim, frames, valid, valid_last
    
    @staticmethod
    def plot_results(time_sim, theta1_sim, theta2_sim, i=None):
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot theta1 and theta2 curves
        ax.plot(time_sim, theta1_sim, label='Theta1')
        ax.plot(time_sim, theta2_sim, label='Theta2')
        # Set labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Joint Angles Over Time')
        # Display legend
        ax.legend()
        # Return the figure object
        return fig, ax

if __name__ == "__main__":
    
    # Load the Ancestors Model
    # path = "./models/v2_4/urdf/dog2_4singleLeg_realconstrast.xml"
    # experiment_instance = MujocoExperiment(path, model_type="real_geom")

    path = "./models/SingleLeg_ideal_for_topology.xml"
    experiment_instance = MujocoExperiment(path, model_type="ideal_geom_swing")

    # Model Parameter Set
    stiffness_MAA, stiffness_BAA = 318.76, 315.8
    l10, l20 = 0.174, 0.2562
    damping_MAA, damping_BAA = 11.34, 10.9
    s1, s2 = 0.000654, 0.000637
    s3 = 1
    c1_thigh, c2_calf = 0.03*0, 0.03*0
    P1 = 0/s1#50*1e3
    P2 = 0
    P3 = 0
    P1_prime = 10/s1#*1e3
    P2_prime = 5/s2#*1e3
    P3_prime = 1
    
    exp_num = 100
    np.random.seed(0)
    tic = time.time()

    params = {
        'stiffness_MAA': stiffness_MAA,
        'stiffness_BAA': stiffness_BAA,
        'l10': l10,
        'l20': l20,
        'damping_MAA': damping_MAA,
        'damping_BAA': damping_BAA,
        's1': s1,
        's2': s2,
        's3': s3,
        'c1_thigh': c1_thigh,
        'c2_calf': c2_calf,
        'P1': P1,
        'P2': P2,
        'P3': P3,
        'P1_prime': P1_prime,
        'P2_prime': P2_prime,
        'P3_prime': P3_prime
    }

    # Run the Experiments
    time_step = 10
    duration_exp = 40  # seconds
    framerate = 60
    # time_sim, theta1_sim, theta2_sim, frames, valid, valid_last, valve1, valve2 = experiment_instance.run_with_valve_dynamics(params, time_step=time_step, duration=duration_exp, ifrender=False)
    time_sim, theta1_sim, theta2_sim, theta3_sim, frames, valid, valid_last = experiment_instance.run(params, time_step=time_step, duration=duration_exp, ifrender=True)

    # Show video
    # media.show_video(frames, fps=framerate)
    media.write_video("./log/temp_experiment_valve_dynamics_temp0717.mp4", frames, fps=framerate)

    # Plot Results with dual y-axis for pressure states
    fig, ax1 = plt.subplots(figsize=(10, 5))
    # Plot theta1, theta2, and theta3 curves
    ax1.plot(time_sim, theta1_sim, label='Theta1', color='tab:blue')
    ax1.plot(time_sim, theta2_sim, label='Theta2', color='tab:orange')
    ax1.plot(time_sim, theta3_sim, label='Theta3', color='tab:red')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (radians)')
    ax1.set_title('Joint Angles and Valve Pressures Over Time')
    ax1.legend(loc='upper left')

    # Create a second y-axis
    # ax2 = ax1.twinx()
    # ax2.plot(time_sim, valve1, label='Valve1 Pressure', color='tab:green', linestyle='--')
    # ax2.plot(time_sim, valve2, label='Valve2 Pressure', color='tab:red', linestyle='--')
    # ax2.set_ylabel('Valve Pressure')
    # ax2.legend(loc='upper right')

    plt.show()

    # Save for cross-validation of the sympy implementation
    import pandas as pd
    data_traj = {
        'time': time_sim,
        'theta1': theta1_sim,
        'theta2': theta2_sim,
        'theta3': theta3_sim,
    }
    df = pd.DataFrame(data_traj)
    df.to_csv('data_mj.csv', index=False)
