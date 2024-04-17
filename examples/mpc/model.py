import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from gym_multirotor.sb3.common.policies import ActorCriticPolicy
from gym_multirotor.sb3 import PPO
from gym_multirotor.sb3.common.env_util import make_vec_env


class NeuralCostMap(nn.Module):
    # outputs of cost map are diagonal entries of Q and P with both state and control included
    def __init__(self, obs_len, state_dim, control_dim, horizon):
        super().__init__()
        self.input_size = obs_len
        self.output_size = 2 * (state_dim + control_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_size),
            F.sigmoid()
        )

    def forward(self, state, input):
        x = torch.cat((state, input), dim=-1)
        return self.network(x)
    

class Critic(nn.Module):
    def __init__(self, obs_len):
        super().__init__()
        self.input_size = obs_len
        self.output_size = 1
        self.network = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_size),
        )


#TODO: implement differentiable mpc
class DifferentiableMPC:
    def __init__(self, state_dim, action_dim, horizon):
        # This would be a complex differentiable MPC module
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon

    def forward(self, state, cost_parameters):
        # Simplified for demonstration; in reality, this would solve an MPC problem
        # Here, just return a random action for the example
        return torch.rand(state.size(0), self.action_dim), torch.rand(state.size(0), 1)  # Dummy action and cost





class ACMPC():
    def __init__(self, obs_len, state_dim, control_dim):
        # Actor, policy, neural cost map
        value_input_size = obs_len
        value_output_size = 2 * (state_dim + control_dim)
        neural_cost_map = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_size),
            F.sigmoid(),
            DifferentiableMPC()
        )

        # Critic, value
        critic_input_size = obs_len
        critic_output_size = 1
        critic = nn.Sequential(
            nn.Linear(critic_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, critic_output_size),
        )

        actor_critic = PPO(
            policy=ActorCriticPolicy,
            env=make_vec_env('QuadrotorPlusHoverEnv-v0', n_envs=4), 
            verbose=1,
            policy_kwargs = {
                'model_arch': [neural_cost_map, critic]
            },
            seed=0
        )

        diff_mpc = DifferentiableMPC()