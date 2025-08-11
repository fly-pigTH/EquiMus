# Ball kicking environment for reinforcement learning using our model in MuJoCo

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

        n_act = 2
        self.action_space = spaces.Box(low=-50, high=50, shape=(n_act,), dtype=np.float64)

        obs_dim = 4 + 3 + 3  # joint position & joint velocity, ball position, ball velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)

        self.renderer = None  # will be created when rendering

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # set the control inputs
        self.data.ctrl[:2] = action[0]
        self.data.ctrl[2:] = action[-1]

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = False
        info = {}

        return obs, reward, bool(terminated), truncated, info

    def _get_obs(self):
        joint_qpos = self.data.qpos[:2]
        joint_qvel = self.data.qvel[:2]

        ball_pos = self.data.body("ball").xpos
        ball_vel = self.data.body("ball").cvel[:3]

        return np.concatenate([joint_qpos, joint_qvel, ball_pos, ball_vel])

    def _compute_reward(self):
        ball_pos = self.data.body("ball").xpos
        ball_vel = self.data.body("ball").cvel[:3]
        leg_tip = self.data.geom("_foot_geom").xpos

        # 1. Reward for approaching the ball (early stage guidance)
        dist_to_ball = np.linalg.norm(leg_tip - ball_pos)
        proximity_reward = 1.0 / (1.0 + dist_to_ball)  # Higher when closer

        # 2. Contact bonus (can be combined with contact detection)
        contact_bonus = 0.0
        if self._check_foot_hits_ball():
            contact_bonus = 2.0

        # 3. Reward for ball velocity in target direction (main objective)
        target_dir = np.array([0.0, -1.0, 0.05])  # Change to your desired kick direction (normalized)
        direction_reward = np.dot(ball_vel, target_dir)

        # 4. Ball speed reward (regardless of direction)
        speed_reward = np.linalg.norm(ball_vel)

        # 5. Control cost penalty (to prevent excessive movement)
        effort_penalty = 0.001 * np.sum(np.square(self.data.ctrl))

        # Total reward
        reward = (
            2.0 * proximity_reward +   # Approach ball
            2.0 * contact_bonus +      # Hit ball
            5.0 * direction_reward +   # Kick accurately
            1.0 * speed_reward -       # Kick fast
            effort_penalty             # Control penalty
        )
        return reward
        
    def _check_foot_hits_ball(self):
        # Get geom IDs for detection (note: geom, not body)
        foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "_foot_geom")
        ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geoms = {contact.geom1, contact.geom2}
            if foot_geom_id in geoms and ball_geom_id in geoms:
                return True  # Collision detected
        return False

    def _check_terminated(self):
        # Terminate based on ball position or simulation instability
        ball_pos = self.data.body("ball").xpos
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
