from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import List, Optional, Type

import ray
from ray import tune
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.ddpg import TD3Trainer
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.agents.callbacks import DefaultCallbacks

import gym
from gym import spaces
import rlbench.gym
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
# from rlbench.tasks import ReachMovingTarget as task_nn
from rlbench.tasks import ReachTarget as task_nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from rlbench_custom_env import RLBenv_phase1, RLBenv_phase2, RLBenv_phase3, RLBenv_phase4, RLBenv_phase5


class TorchCustomModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.fclayer = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
    
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        out = self.fclayer(obs)
        return out, []

# class episode_callback(DefaultCallbacks):

def curriculum_train(config, reporter):
    
    agent1 = TD3Trainer(env='RLBench_phase1', config = config)
    for _ in range(5):
        result = agent1.train()
        result["phase"] = 1
        reporter(**result)
    # agent4 = TD3Trainer(env='RLBench_phase4', config = config)
    # for _ in range(10):
    #     result = agent4.train()
    #     result["phase"] = 4
    #     reporter(**result)

    while result["episode_reward_mean"] < 7:
        result = agent1.train()
        result["phase"] = 1
        reporter(**result)
        state = agent1.save()
    agent1.stop()

    agent2 = TD3Trainer(env='RLBench_phase2', config = config)
    agent2.restore(state)
    while result["episode_reward_mean"] < 7.5:
        result = agent2.train()
        result["phase"] = 2
        reporter(**result)
        state = agent2.save()
    agent2.stop()

    agent3 = TD3Trainer(env='RLBench_phase3', config = config)
    agent3.restore(state)
    while result["episode_reward_mean"] < 8:
        result = agent3.train()
        result["phase"] = 3
        reporter(**result)
        state = agent3.save()
    agent3.stop()

    agent4 = TD3Trainer(env='RLBench_phase4', config = config)
    agent4.restore(state)
    while result["episode_reward_mean"] < 9:
        result = agent4.train()
        result["phase"] = 4
        reporter(**result)
        state = agent4.save()
    agent4.stop()

    agent5 = TD3Trainer(env='RLBench_phase5', config = config)
    agent5.restore(state)
    while result["episode_reward_mean"] < 10:
        result = agent5.train()
        result["phase"] = 5
        reporter(**result)
        agent5.save()
    agent5.stop()


if __name__ == '__main__':
    ray.init(num_gpus=1)

    register_env('RLBench_phase1', lambda _: RLBenv_phase1())
    register_env('RLBench_phase2', lambda _: RLBenv_phase2())
    register_env('RLBench_phase3', lambda _: RLBenv_phase3())
    register_env('RLBench_phase4', lambda _: RLBenv_phase4())
    register_env('RLBench_phase5', lambda _: RLBenv_phase5())

    ModelCatalog.register_custom_model(
        "my_custom_model", TorchCustomModel
    )

    config = {
        'framework': 'torch',
        # 'callbacks': 'episode_callback'
        #=== Model ===
        'model': {
            'custom_model': 'my_custom_model',
            'custom_model_config': {},
        },
        # 'actor_hiddens': [256, 256],
        # 'actor_hidden_activation': 'relu',
        # 'critic_hiddens': [256, 256],
        # 'critic_hidden_activation': 'relu',
        #=== Exploration ===
        # 'learning_starts' : 0,
        'exploration_config': {
        },
        'timesteps_per_iteration': 1000,
        #=== Relay buffer ===
        'buffer_size': 15000,
        #=== Optimization ===
        'critic_lr': 1e-3,
        'actor_lr': 1e-3,
        #=== Parallelism ===
        'num_cpus_per_worker': 1,
        'num_workers': 4,
        'num_gpus': 1,
        #=== Evaluation ===
    }

    tune.run(
        curriculum_train,
        # stop = {'training_iteration': 100000},
        checkpoint_freq=1,
        local_dir = '/home/gpu/Workspace/Sangbeom/ray_results',
        # restore="/home/sangbeom/checkpoint_439/checkpoint-439",
        config=config,
        resources_per_trial = {
            "cpu": 1,
            "gpu": 1,
            "extra_cpu": 4,
        },
    )

