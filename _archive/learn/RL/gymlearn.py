import gymnasium as gym

# Create the Half-Cheetah environment
env = gym.make("HalfCheetah-v5")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment for parallel simulation
env = make_vec_env("HalfCheetah-v5", n_envs=4)

# Define the RL model
model = PPO("MlpPolicy", env, verbose=1)

# Optional: Add evaluation callback
eval_env = gym.make("HalfCheetah-v5")
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=10000,
                             deterministic=True, render=False)

# Train the model
model.learn(total_timesteps=1_0_000, callback=eval_callback)

# Save the model
model.save("ppo_half_cheetah")


# evaluate
from stable_baselines3.common.evaluation import evaluate_policy

# Load the trained model
model = PPO.load("ppo_half_cheetah")

# Evaluate the policy
env = gym.make("HalfCheetah-v5")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")


## Visialize the agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()