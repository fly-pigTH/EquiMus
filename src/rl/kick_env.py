import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

class KickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, xml_path="kick_ball.xml", render_mode=None, frame_skip=10):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.frame_skip = frame_skip

        # TODO: 根据你要控制的关节数量设置动作维度
        n_act = 2
        self.action_space = spaces.Box(low=-50, high=50, shape=(n_act,), dtype=np.float64)

        # TODO: 根据你观察的内容设定 observation 维度（joint, ball等）
        obs_dim = 4 + 3 + 3  # joint position, joint velocity, ball position, ball velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)

        self.renderer = None  # will be created when rendering

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # TODO: 如有球或机器人初始姿态设置，可添加这里
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # TODO: 根据实际 actuator 映射 action（是否乘上最大控制值等）
        self.data.ctrl[:2] = action[0]  # 假设前两个 actuator 控制关节
        self.data.ctrl[2:] = action[-1]  # 假设前两个 actuator 控制关节

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = False
        info = {}

        return obs, reward, bool(terminated), truncated, info

    def _get_obs(self):
        # TODO: 修改 joint 范围为你实际控制的部分
        joint_qpos = self.data.qpos[:2]
        joint_qvel = self.data.qvel[:2]

        # TODO: 替换 "ball" 为你模型中球的 body 名称
        ball_pos = self.data.body("ball").xpos
        ball_vel = self.data.body("ball").cvel[:3]

        return np.concatenate([joint_qpos, joint_qvel, ball_pos, ball_vel])

    def _compute_reward(self):
        # ===== 获取关键信息 =====
        ball_pos = self.data.body("ball").xpos
        ball_vel = self.data.body("ball").cvel[:3]
        leg_tip = self.data.geom("_foot_geom").xpos  # 假设脚部的 geom 名称为 "foot_geom"

        # ===== 1. 靠近球的奖励（early stage 引导）=====
        dist_to_ball = np.linalg.norm(leg_tip - ball_pos)
        proximity_reward = 1.0 / (1.0 + dist_to_ball)  # 越近越高

        # ===== 2. 接触奖励（可配合 contact 检测）=====
        contact_bonus = 0.0
        if self._check_foot_hits_ball():
            contact_bonus = 2.0

        # ===== 3. 球向目标方向的速度奖励（主要目标）=====
        target_dir = np.array([0.0, -1.0, 0.05])  # TODO: 修改为你想踢的方向（已单位化）
        direction_reward = np.dot(ball_vel, target_dir)

        # ===== 4. 球速奖励（不考虑方向）=====
        speed_reward = np.linalg.norm(ball_vel)

        # ===== 5. 控制代价惩罚（防止乱动）=====
        effort_penalty = 0.001 * np.sum(np.square(self.data.ctrl))

        # ===== 汇总总奖励 =====
        reward = (
            2.0 * proximity_reward +   # 靠近球
            2.0 * contact_bonus +      # 踢到球
            5.0 * direction_reward +   # 踢的准
            1.0 * speed_reward -       # 踢得快
            effort_penalty             # 控制惩罚
        )
        return reward
        
    def _check_foot_hits_ball(self):
        # 获取需要检测的 geom ID（注意是 geom，不是 body）
        foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "_foot_geom")
        ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geoms = {contact.geom1, contact.geom2}
            if foot_geom_id in geoms and ball_geom_id in geoms:
                return True  # 检测到碰撞
        return False


    def _check_terminated(self):
        # TODO: 可根据球位置或仿真不稳定判断终止
        ball_pos = self.data.body("ball").xpos
        # print(ball_pos[0] > 2.0 or np.any(np.isnan(ball_pos)))
        # print("-----")
        return ball_pos[0] > 2.0 or np.any(np.isnan(ball_pos))

    def render(self):
        with mujoco.Renderer(self.model, width=640, height=480) as renderer:
            renderer.update_scene(self.data, camera="closeup")
            frame = renderer.render()
        return frame

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None