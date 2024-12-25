import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# 创建 Hopper 跳跃环境
env = gym.make('Hopper-v4')

# 初始化 PPO 模型
model = PPO("MlpPolicy", env, verbose=1)

# 开始训练
print("开始训练...")
model.learn(total_timesteps=100000)  # 设置训练步数
print("训练完成！")

# 保存模型
model.save("ppo_hopper_jump")
print("模型保存成功！")

# 评估模型
print("评估模型...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"平均奖励: {mean_reward} ± {std_reward}")