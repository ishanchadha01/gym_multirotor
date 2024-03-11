"""
Use terminal to run this code.

python play_multirotor.py
"""

import gymnasium as gym
import numpy as np
import gym_multirotor
import time


def main(env):
    render = True
    # env_ = env.env.env.env
    # # ----- Environment Info ------------------------
    # obs_dimensions = env.observation_space.shape[0]
    # print("Observation dimensions:", obs_dimensions)

    # action_dimensions = env.action_space.shape[0]
    # print("Action dimensions:", action_dimensions)

    # min_action = env.action_space.low
    # print("Min. action:", min_action)

    # max_action = env.action_space.high
    # print("Max. action:", max_action)

    # print("Actuator_control:", type(env.model.actuator_ctrlrange))
    # print("actuator_forcerange:", env.model.actuator_forcerange)
    # print("actuator_forcelimited:", env.model.actuator_forcelimited)
    # print("actuator_ctrllimited:", env.model.actuator_ctrllimited)
    # # --------------------------------------------

    ob = env.reset()
    done = False
    if render:
        env.render()

    for t in range(10000):
        if render:
            env.render()

        # action = np.array([0, 0, 0, 0])
        action = env.action_space.sample() # this samples random action, we want ppo

        ob, reward, terminated, truncated, info = env.step(action)

        if done:
            ob, info = env.reset()
        # print(info)
        # print(ob)
        time.sleep(1.0)

    env.close()


if __name__ == "__main__":
    env = gym.make('QuadrotorPlusHoverEnv-v0', render_mode='human')
    # env = gym.make("Ant-v4", render_mode="human")
    # env = gym.make("LunarLander-v2",render_mode="human")
    # env = gym.make('TiltrotorPlus8DofHoverEnv-v0')
    main(env)
