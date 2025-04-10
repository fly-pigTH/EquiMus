import mediapy as media
from kick_env import KickEnv
from stable_baselines3 import PPO
from tqdm import tqdm
import mujoco
dir_name = "PPO_27"

env = KickEnv(xml_path="models/v2_4/urdf/dog2_4singleLeg_realconstrast_kickball.xml", render_mode="rgb_array")
model = PPO.load(f"ppo_kick_tensorboard/{dir_name}/ppo_kick_model")

obs, _ = env.reset()
frames = []
action_list = []
# 裸用，为了加速
with mujoco.Renderer(env.model, width=640, height=480) as renderer:
    for _ in tqdm(range(4*1000)):
        action, _ = model.predict(obs, deterministic=True)
        # random policy
        # action = env.action_space.sample()
        action_list.append(action)
        obs, _, done, _, _ = env.step(action)       # consider skip of frames
        if len(frames) < env.data.time * 120:   # assume the simulation is running much faster than the rendering        
                renderer.update_scene(env.data, camera="closeup")
                frame = renderer.render()
                frames.append(frame)
        if done:
            break

# save action_list
import numpy as np
np.save(f"ppo_kick_tensorboard/{dir_name}/kick_action.npy", action_list)

media.write_video(f"ppo_kick_tensorboard/{dir_name}/kick_result.mp4", frames, fps=120)
env.close()
