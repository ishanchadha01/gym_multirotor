"""
Use terminal to run this code.

python play_multirotor.py
"""

import gymnasium as gym
import numpy as np
from gym_multirotor.sb3 import PPO
from gym_multirotor.sb3.common.env_util import make_vec_env


def main_old():
    env = gym.make('QuadrotorPlusHoverEnv-v0', render_mode='human')
    # env = gym.make("Ant-v4", render_mode="human")
    # env = gym.make("LunarLander-v2",render_mode="human")
    # env = gym.make('TiltrotorPlus8DofHoverEnv-v0')
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
        action = env.action_space.sample()

        ob, reward, terminated, truncated, info = env.step(action)

        if done:
            ob, info = env.reset()
        print(info)
        print(ob)

    env.close()


def main():
    vec_env = make_vec_env('QuadrotorPlusHoverEnv-v0', n_envs=4)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("quadplus")

    del model # remove to demonstrate saving and loading

    model = PPO.load("quadplus")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    # main_old()
    main()

    
    

    
    
