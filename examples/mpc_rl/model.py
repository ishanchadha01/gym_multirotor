import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from gym_multirotor.sb3.common.policies import ActorCriticPolicy
from gym_multirotor.sb3 import PPO
from gym_multirotor.sb3.common.env_util import make_vec_env

import cv2
import json
from gym_multirotor.mpc.mpc.mpc import QuadCost, LinDx, MPC


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


class MPCActor(nn.Module):
    def __init__(self, state_dim, control_dim, obs_len):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.obs_len = obs_len

        value_input_size = obs_len
        value_output_size = 2 * (state_dim + control_dim)

        self.linear1 = nn.Linear(value_input_size, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, value_output_size)
        self.diff_mpc = DifferentiableMPC(state_dim, control_dim)
        self.out_size = control_dim

    def forward(self, obs, x_init):
        y = self.linear1(obs)
        y = F.relu(y)
        y = self.linear2(y)
        y = F.relu(y)
        y = self.linear3(y)
        y = F.sigmoid(y)

        Q = y[:, :self.state_dim]
        p = y[:, self.state_dim:]
        u = self.diff_mpc(x_init, Q, p)
        return u


class MPCCritic(nn.Module):
    def __init__(self, obs_len):
        super().__init__()
        critic_input_size = obs_len
        critic_output_size = 1
        self.feats = nn.Sequential(
            nn.Linear(critic_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, critic_output_size),
        )
        self.out_size = critic_output_size

    def forward(self, obs):
        return self.feats(obs)


#TODO: implement differentiable mpc
class DifferentiableMPC(nn.Module):
    def __init__(self, state_dim, control_dim, horizon=5, device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        # self.Q = torch.eye(2*(state_dim+control_dim)).to(device)
        # self.p = torch.ones(2*(state_dim+control_dim)).to(device)
        self.T = horizon
        self.A = (torch.eye(state_dim) + .1*torch.randn(state_dim, state_dim)).to(device).requires_grad_()
        self.B = torch.randn(state_dim, control_dim).to(device).requires_grad_()

    def forward(self, x_init, Q, p): # TODO: how to get x_init?
        num_batches = Q.shape[0]
        F = torch.cat((self.A, self.B), dim=1).unsqueeze(0).unsqueeze(0).repeat(self.T, num_batches, 1, 1)
        x_pred, u_pred, objs_pred = MPC(
            self.state_dim, self.control_dim, self.T,
            u_lower=None, u_upper=None, u_init=None,
            lqr_iter=100,
            verbose=-1,
            exit_unconverged=False,
            detach_unconverged=False,
            n_batch=num_batches,
        )(x_init, QuadCost(Q, p), LinDx(F))
        return u_pred


class ACMPC():
    def __init__(self, obs_len, state_dim, control_dim):

        # Create env
        self.env = make_vec_env('QuadrotorPlusHoverEnv-v0', n_envs=4)

        # Actor, policy, neural cost map
        actor = MPCActor(state_dim=state_dim, control_dim=control_dim, obs_len=obs_len)

        # Critic, value, TODO: make this separate module?
        critic = MPCCritic(obs_len=obs_len)

        self.actor_critic = PPO(
            policy=ActorCriticPolicy,
            env=self.env, 
            verbose=1,
            policy_kwargs = {
                'model_arch': [actor, critic]
            },
            seed=0
        )
    
    def train(self):
        fp = "mpc_output_nowind.json"
        data = []
        for i in range(100):
            self.actor_critic.learn(total_timesteps=1e5, log_interval=1)
            self.actor_critic.save("quadplus_mpc_nowind")
            obs = self.reset()
            
            # write to json file
            print(self.actor_critic.log_outputs)
            iter_data = {key: float(value) for key, value in self.actor_critic.log_outputs.items()}
            data.append(iter_data)
            with open(fp, 'w') as f:
                json.dump(data, f, indent=4)

            imgs = []
            for j in range(100):
                action, _states = self.actor_critic.predict(obs)
                obs, rewards, dones, info = self.env.step(action)
                bigimg = self.env.render("rgb_array")
                imgs.append(bigimg[:,:,::-1])
            create_video(imgs, f"mpc_nowind_vids/vid_{i:04d}.mp4", fps=24)
        f.close()