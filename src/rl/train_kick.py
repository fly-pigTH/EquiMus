from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from kick_env import KickEnv
from stable_baselines3.common.monitor import Monitor
import os

def make_env(xml_path):
    def _init():
        env = KickEnv(xml_path=xml_path, render_mode=None)
        # ✅ 可选：添加监控器，记录日志
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    # ✅ TODO: 修改你的模型路径
    xml_path = "models/v2_4/urdf/dog2_4singleLeg_realconstrast_kickball.xml"

    # ✅ 启动 4 个并行环境（可改成 8、16，根据你的 CPU）
    num_envs = 16
    env_fns = [make_env(xml_path) for _ in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # ✅ 可选：验证你的环境是否符合规范 (only one)
    check_env(make_env(xml_path)(), warn=True)

    # ✅ 创建 PPO 模型，建议用 GPU（device="cuda"）
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=4096,          # 每次 rollout 的长度（越大越快）
        batch_size=1024,
        n_epochs=10,
        device="cuda",         # 如果有 GPU，强烈建议加速
        tensorboard_log="./ppo_kick_tensorboard",
        policy_kwargs=dict(log_std_init=0.0)  # ✅ 默认是 0.0，建议设置成 1.0 ~ 2.0
    )

    # ✅ 启动训练
    model.learn(total_timesteps=50_000_000)
    # 获取 tensorboard_log 实际路径
    save_path = model.logger.dir  # e.g., 'log/PPO_MyEnv_20250409_123456/'

    # 保存模型
    model.save(os.path.join(save_path, "ppo_kick_model"))

    print(f"✅ 模型已保存到: {save_path}")