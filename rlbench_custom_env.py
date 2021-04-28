import gym
from gym import spaces
import rlbench.gym
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
# from rlbench.tasks import ReachMovingTarget as task_nn
from rlbench.tasks import ReachTarget as task_nn

import numpy as np

class RLBenv_phase1(gym.Env):

    def __init__(self):
        self.episode_length = 300
        self.ts = 0
        self.num_error = 0
        obs_config = ObservationConfig(
            joint_velocities = False,
            joint_positions = True,
            joint_forces = False,
            gripper_open = False,
            gripper_pose = False,
            task_low_dim_state = True,
        )
        obs_config.set_all_high_dim(False)
        action_mode = ActionMode(ArmActionMode.DELTA_JOINT_POSITION)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True
        )
        self.env.launch()
        self.task = self.env.get_task(task_nn)
        _, obs = self.task.reset()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_size,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
    
    def action_input(self, action):
        action = action * 0.05
        return action

    def _extract_obs (self, obs):
        return obs.get_low_dim_data()

    def reset(self):
        descriptions, obs = self.task.reset()
        self.ts = 1
        return self._extract_obs(obs)

    def step(self, action):
        action = self.action_input(action)
        obs, reward, done = self.task.step(action)
        if reward == 10:
            print ('========success=========')
            reward = 10
        self.ts = self.ts + 1
        if self.ts > self.episode_length:
            return self._extract_obs(obs), reward, True, {}
        else:
            return self._extract_obs(obs), reward, done, {}
            
    def close(self):
        self.env.shutdown()

class RLBenv_phase2(gym.Env):

    def __init__(self):
        self.episode_length = 300
        self.ts = 0
        self.num_error = 0
        obs_config = ObservationConfig(
            joint_velocities = False,
            joint_positions = True,
            joint_forces = False,
            gripper_open = False,
            gripper_pose = False,
            task_low_dim_state = True,
        )
        obs_config.set_all_high_dim(False)
        action_mode = ActionMode(ArmActionMode.DELTA_JOINT_POSITION)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True
        )
        self.env.launch()
        self.task = self.env.get_task(task_nn)
        _, obs = self.task.reset()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_size,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
    
    def action_input(self, action):
        action = action * 0.05
        return action

    def _extract_obs (self, obs):
        return obs.get_low_dim_data()

    def reset(self):
        descriptions, obs = self.task.reset()
        self.ts = 1
        return self._extract_obs(obs)

    def step(self, action):
        action = self.action_input(action)
        obs, reward, done = self.task.step(action)
        
        if reward == 10:
            print ('========success=========')
            reward = 10
        # elif reward <= -1:
        #     reward = -1
        # elif reward <= -0.75:
        #     reward = -0.75
        # elif reward <= -0.5:
        #     reward = -0.5
        # elif reward <= -0.25:
        #     reward = -0.25
        # elif reward <= 0:
        #     reward = 0
        elif self.ts % 10 == 0:
            reward = reward
        else:
            reward = 0

        self.ts = self.ts + 1
        if self.ts > self.episode_length:
            return self._extract_obs(obs), reward, True, {}
        else:
            return self._extract_obs(obs), reward, done, {}
            
    def close(self):
        self.env.shutdown()

class RLBenv_phase3(gym.Env):

    def __init__(self):
        self.episode_length = 300
        self.ts = 0
        self.num_error = 0
        obs_config = ObservationConfig(
            joint_velocities = False,
            joint_positions = True,
            joint_forces = False,
            gripper_open = False,
            gripper_pose = False,
            task_low_dim_state = True,
        )
        obs_config.set_all_high_dim(False)
        action_mode = ActionMode(ArmActionMode.DELTA_JOINT_POSITION)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True
        )
        self.env.launch()
        self.task = self.env.get_task(task_nn)
        _, obs = self.task.reset()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_size,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
    
    def action_input(self, action):
        action = action * 0.05
        return action

    def _extract_obs (self, obs):
        return obs.get_low_dim_data()

    def reset(self):
        descriptions, obs = self.task.reset()
        self.ts = 1
        return self._extract_obs(obs)

    def step(self, action):
        action = self.action_input(action)
        obs, reward, done = self.task.step(action)
        if reward == 10:
            print ('========success=========')
            reward = 10
        # elif reward <= -0.5:
        #     reward = -0.5
        # elif reward <= -0.25:
        #     reward = -0.25
        # elif reward <= 0:
        #     reward = 0
        elif self.ts % 25 == 0:
            reward = reward
        else:
            reward = 0
        self.ts = self.ts + 1
        if self.ts > self.episode_length:
            return self._extract_obs(obs), reward, True, {}
        else:
            return self._extract_obs(obs), reward, done, {}
            
    def close(self):
        self.env.shutdown()

class RLBenv_phase4(gym.Env):

    def __init__(self):
        self.episode_length = 300
        self.ts = 0
        self.num_error = 0
        obs_config = ObservationConfig(
            joint_velocities = False,
            joint_positions = True,
            joint_forces = False,
            gripper_open = False,
            gripper_pose = False,
            task_low_dim_state = True,
        )
        obs_config.set_all_high_dim(False)
        action_mode = ActionMode(ArmActionMode.DELTA_JOINT_POSITION)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True
        )
        self.env.launch()
        self.task = self.env.get_task(task_nn)
        _, obs = self.task.reset()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_size,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
    
    def action_input(self, action):
        action = action * 0.05
        return action

    def _extract_obs (self, obs):
        return obs.get_low_dim_data()

    def reset(self):
        descriptions, obs = self.task.reset()
        self.ts = 1
        return self._extract_obs(obs)

    def step(self, action):
        action = self.action_input(action)
        obs, reward, done = self.task.step(action)

        if reward == 10:
            print ('========success=========')
            reward = 10
        # elif reward <= -0.25:
        #     reward = -0.25
        # elif reward <= 0:
        #     reward = 0
        elif self.ts % 50 == 0:
            reward = reward
        else:
            reward = 0

        self.ts = self.ts + 1
        if self.ts > self.episode_length:
            return self._extract_obs(obs), reward, True, {}
        else:
            return self._extract_obs(obs), reward, done, {}
            
    def close(self):
        self.env.shutdown()

class RLBenv_phase5(gym.Env):

    def __init__(self):
        self.episode_length = 300
        self.ts = 0
        self.num_error = 0
        obs_config = ObservationConfig(
            joint_velocities = False,
            joint_positions = True,
            joint_forces = False,
            gripper_open = False,
            gripper_pose = False,
            task_low_dim_state = True,
        )
        obs_config.set_all_high_dim(False)
        action_mode = ActionMode(ArmActionMode.DELTA_JOINT_POSITION)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True
        )
        self.env.launch()
        self.task = self.env.get_task(task_nn)
        _, obs = self.task.reset()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_size,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
    
    def action_input(self, action):
        action = action * 0.05
        return action

    def _extract_obs (self, obs):
        return obs.get_low_dim_data()

    def reset(self):
        descriptions, obs = self.task.reset()
        self.ts = 1
        return self._extract_obs(obs)

    def step(self, action):
        action = self.action_input(action)
        obs, reward, done = self.task.step(action)
        if reward == 10:
            print ('========success=========')
            reward = 10
        elif reward <= 0:
            reward = 0

        self.ts = self.ts + 1
        if self.ts > self.episode_length:
            return self._extract_obs(obs), reward, True, {}
        else:
            return self._extract_obs(obs), reward, done, {}
            
    def close(self):
        self.env.shutdown()
