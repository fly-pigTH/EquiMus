from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

# Basic Path
import rootpath
from pathlib import Path
import numpy as np

# need to tackle specifically, for compatiblity (module and main)
import sys
import rootpath
ROOT_DIR = rootpath.detect()   # Get the root directory of the project (.git)
sys.path.append(str(Path(ROOT_DIR) / "src" / "application" / "RL_BallKicking" / "scripts"))
from kick_env import KickEnv


def make_env(xml_path):
    def _init():
        env = KickEnv(xml_path=xml_path, render_mode=None)
        env = Monitor(env)
        return env
    return _init

def train():
    ROOT_DIR = rootpath.detect()   # Get the root directory of the project (.git)
    CURRENT_DIR = Path(__file__).resolve().parent
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"CURRENT_DIR: {CURRENT_DIR}")
    xml_path = str(Path(ROOT_DIR) / "models" / "v2_4" / "urdf" / "dog2_4singleLeg_realconstrast_kickball.xml")

    timesteps_schedule = np.array([0.4*1e6, 4*1e6, 40*1e6])

    for stage, steps in enumerate(timesteps_schedule):

        # Launch 16 parallel environments
        num_envs = 16
        env_fns = [make_env(xml_path) for _ in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")

        # Optional: Check if your environment conforms to the standard (only one)
        check_env(make_env(xml_path)(), warn=True)

        # Create PPO model, recommend using GPU (device="cuda")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            n_steps=4096,          # Length of each rollout (larger is faster)
            batch_size=1024,
            n_epochs=10,
            device="cpu",         # Strongly recommend using GPU if available
            tensorboard_log=CURRENT_DIR.parent / "log" / "ppo_kick_tensorboard",
            policy_kwargs=dict(log_std_init=0.0),  # Default is 0.0
            seed=42,  # For reproducibility
        )
        print(f"Stage {stage}: Training for {timesteps_schedule[stage]} timesteps")
        model.learn(total_timesteps=steps, tb_log_name=f"ppo_kick_log_{stage}")

        # Save a checkpoint after each stage
        checkpoint_path = CURRENT_DIR.parent / "data" / "model" / f"ppo_kick_model_{timesteps_schedule[stage]}.zip"
        model.save(checkpoint_path)
        print(f"Stage {stage} done. Model saved to {checkpoint_path}")

if __name__ == "__main__":
    train()