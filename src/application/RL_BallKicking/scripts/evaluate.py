import mediapy as media
from kick_env import KickEnv
from stable_baselines3 import PPO
from tqdm import tqdm
import mujoco
import numpy as np
import rootpath
from pathlib import Path

def evaluate():
    model_list = [
        "ppo_kick_model_400000.0.zip", 
        "ppo_kick_model_4000000.0.zip",
        "ppo_kick_model_40000000.0.zip"
    ]

    iteration_list = [400000, 4000000, 40000000]

    for model_name, iterations in zip(model_list, iteration_list):
        print(f"Loading model: {model_name} for {iterations} iterations")

        ROOT_DIR = Path(rootpath.detect())   # Get the root directory of the project (.git)
        CURRENT_DIR = Path(__file__).resolve().parent
        MODEL_DIR = CURRENT_DIR.parent / "data" / "model"
        LOG_DIR = CURRENT_DIR.parent / "log"
        VIDEO_DIR = CURRENT_DIR.parent / "video"

        XML_PATH = str(ROOT_DIR / "models" / "v2_4" / "urdf" / "dog2_4singleLeg_realconstrast_kickball.xml")
        env = KickEnv(xml_path=XML_PATH, render_mode="rgb_array")
        model = PPO.load(MODEL_DIR / model_name)

        obs, _ = env.reset()
        frames = []
        action_list = []
        # Bare usage, for acceleration
        with mujoco.Renderer(env.model, width=640, height=480) as renderer:
            for _ in tqdm(range(4*1000)):
                action, _ = model.predict(obs, deterministic=True)
                action_list.append(action)
                obs, _, done, _, _ = env.step(action)       # consider skip of frames
                if len(frames) < env.data.time * 120:   # assume the simulation is running much faster than the rendering        
                        renderer.update_scene(env.data, camera="closeup")
                        frame = renderer.render()
                        frames.append(frame)
                if done:
                    break

        # save result: action, video
        # np.save(LOG_DIR / "ppo_kick_tensorboard" / f"kick_action_{iterations}.npy", action_list)
        media.write_video(VIDEO_DIR / f"kick_result_{iterations}.mp4", frames, fps=120)
        env.close()

if __name__ == "__main__":
    evaluate()
   