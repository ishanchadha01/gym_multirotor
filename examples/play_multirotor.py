"""
Use terminal to run this code.

python play_multirotor.py
"""

import gymnasium as gym
import numpy as np
from gym_multirotor.sb3 import PPO
from gym_multirotor.sb3.common.env_util import make_vec_env

import cv2
import json

from mpc_rl.model import ACMPC

def main_old():
    env = gym.make('QuadrotorPlusHoverEnv-v0', render_mode='human')
    # env = gym.make("Ant-v4", render_mode="human")
    # env = gym.make("LunarLander-v2",render_mode="human")
    # env = gym.make('TiltrotorPlus8DofHoverEnv-v0')
    render = False
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


def create_video(imgs, output_path, fps=10):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    # Assume all images are the same size, read the first image to get the size
    test_img = imgs[0]
    size = (test_img.shape[1], test_img.shape[0])
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    # Read each image and write it to the video
    for img in imgs:
        out.write(img)

    # Release everything when job is finished
    out.release()


def main_ppo_nowind():
    vec_env = make_vec_env('QuadrotorPlusHoverEnv-v0', n_envs=32)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    fp = "ppo_output_nowind.json"
    data = []
    for i in range(100):
        model.learn(total_timesteps=1, log_interval=1)
        # model.save("quadplus_ppo_nowind")
        # obs = vec_env.reset()
        
        # # write to json file
        # print(model.log_outputs)
        # iter_data = {key: float(value) for key, value in model.log_outputs.items()}
        # data.append(iter_data)
        # with open(fp, 'w') as f:
        #     json.dump(data, f, indent=4)

        # imgs = []
        # for j in range(100):
        #     action, _states = model.predict(obs)
        #     obs, rewards, dones, info = vec_env.step(action)
        #     bigimg = vec_env.render("rgb_array")
        #     imgs.append(bigimg[:,:,::-1])
        # create_video(imgs, f"ppo_nowind_vids/vid_{i:04d}.mp4", fps=24)
    # f.close()


def main_mpc_nowind():
    obs_len = 18 # 18-dim numpy array of states of environment consisting of (err_x, err_y, err_z, rot_mat(3, 3), vx, vy, vz, body_rate_x, body_rate_y, body_rate_z)
    state_dim = obs_len
    control_dim = 4 # one for each actuator
    model = ACMPC(obs_len=obs_len, state_dim=state_dim, control_dim=control_dim)
    model.train()
    

def main():
    pass


if __name__ == "__main__":
    main_mpc_nowind()

    # state is qpos, qvel, can get with super()._get_obs() or self.mujoco_qpos/qvel
    # action is ..., same as control input
    # obs is 18 dim array consisting of more values like motion error, rotation, etc


    
    

    
    
