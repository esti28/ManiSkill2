from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose

from .base_env import StationaryManipulationEnv


@register_env("Reach-v0", max_episode_steps=200)
class ReachEnv(StationaryManipulationEnv):
    goal_thresh = 0.025
    min_goal_dist = 0.05

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _initialize_task(self, max_trials=100, verbose=False):
        robot_pos = self.tcp.pose.p

        # Sample a goal position far enough from the object
        for i in range(max_trials):
            goal_xy = self._episode_rng.uniform(-0.1, 0.1, [2])
            goal_z = self._episode_rng.uniform(0, 0.5) + robot_pos[2]
            goal_pos = np.hstack([goal_xy, goal_z])
            if np.linalg.norm(goal_pos - robot_pos) > self.min_goal_dist:
                if verbose:
                    print(f"Found a valid goal at {i}-th trial")
                break

        self.goal_pos = goal_pos
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            goal_pos=self.goal_pos,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
            )
        return obs

    def check_goal_reached(self):
        return np.linalg.norm(self.goal_pos - self.tcp.pos.p) <= self.goal_thresh

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[:-2]
        return np.max(np.abs(qvel)) <= thresh

    def evaluate(self, **kwargs):
        is_robot_static = self.check_robot_static()
        is_goal_reached = self.check_goal_reached()
        return dict(
            is_robot_static=is_robot_static,
            is_goal_reached=is_goal_reached,
            success=is_goal_reached and is_robot_static,
        )

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 5
            return reward

        tcp_to_goal_pos = self.goal_pos - self.tcp.pose.p
        tcp_to_goal_dist = np.linalg.norm(tcp_to_goal_pos)
        reaching_reward = 1 - np.tanh(5 * tcp_to_goal_dist)
        reward += reaching_reward

        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 5.0

    def render_rgb_array(self):
        self.goal_site.unhide_visual()
        ret = super().render_rgb_array()
        self.goal_site.hide_visual()
        return ret

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])