# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from gym_multirotor.sb3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
