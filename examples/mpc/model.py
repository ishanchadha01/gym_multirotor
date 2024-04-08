import torch
import torch.nn as nn
from torch.distributions import Normal

from gym_multirotor.sb3.common.policies import ActorCriticPolicy


class NeuralCostMap(nn.Module):
    def __init__(self, state_dim, action_dim, horizon):
        super().__init__()
        self.horizon = horizon
        self.network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, horizon * (state_dim + action_dim) * 2)
        )

    def forward(self, state):
        return self.network(state).split(self.horizon, dim=1)


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


class ACMPCPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, horizon=10):
        super().__init__(observation_space, action_space)
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]

        self.cost_map = NeuralCostMap(state_dim, action_dim, horizon)
        self.mpc = DifferentiableMPC(state_dim, action_dim, horizon)

    def _predict(self, observation, deterministic=False):
        cost_params = self.cost_map(observation)
        action, _ = self.mpc.forward(observation, cost_params)
        return action

    #TODO: evaluate actions
    def evaluate_actions(self, obs, actions):
        # Here you would include the logic to evaluate actions using the MPC and the neural cost map
        # For simplicity, we just call the super method
        return super().evaluate_actions(obs, actions)

# Usage
observation_space = torch.randn(10)  # Dummy observation space
action_space = torch.randn(2)  # Dummy action space

ac_mpc_policy = ACMPCPolicy(observation_space, action_space)

# Example forward pass
obs = torch.randn(1, 10)  # Dummy observation
action = ac_mpc_policy._predict(obs)
