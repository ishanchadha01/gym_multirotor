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
        # self.diff_mpc = DifferentiableMPC(state_dim, control_dim)
        self.diff_mpc = DiffMPC2(state_dim, control_dim)
        self.out_size = control_dim

    def forward(self, obs, x_init):
        y = self.linear1(obs)
        y = F.relu(y)
        y = self.linear2(y)
        y = F.relu(y)
        y = self.linear3(y)
        y = F.sigmoid(y)

        Q = y[:, :(self.state_dim+self.control_dim)]
        p = y[:, (self.state_dim+self.control_dim):]
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


class DifferentiableMPC(nn.Module):
    def __init__(self, state_dim, control_dim, horizon=1, device='cpu'):
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
        num_state_ctrl = Q.shape[-1]
        Q_eye = torch.eye(num_state_ctrl, device=Q.device, dtype=Q.dtype)
        Q_eye = Q_eye.view(1, num_state_ctrl, num_state_ctrl).expand(*Q.shape, num_state_ctrl) # [n_batch, n_tau, n_tau]
        Q_expanded = Q.unsqueeze(-1).expand_as(Q_eye)
        Q = Q_expanded * Q_eye
        x_init = x_init.float()

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


class DiffMPC2(nn.Module):
    def __init__(self, state_dim, control_dim, horizon=1, device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim

    def forward(self, x_init, Q, p):
        # x_init = x.clone().detach().float().requires_grad_(True)  # Assume x_init should not receive gradients

        
        # x_init = x_init.float()
        torch.set_grad_enabled(True)
        num_batches = Q.shape[0]
        num_state_ctrl = Q.shape[-1]
        Q_eye = torch.eye(num_state_ctrl, device=Q.device, dtype=Q.dtype)
        Q_eye = Q_eye.view(1, num_state_ctrl, num_state_ctrl).expand(*Q.shape, num_state_ctrl) # [n_batch, n_tau, n_tau]
        Q_expanded = Q.unsqueeze(-1).expand_as(Q_eye)
        Q = Q_expanded * Q_eye
        Q = Q.requires_grad_(True)
        p = p.requires_grad_(True)

        x_init = x_init.float().requires_grad_(True)
        u_init = torch.randn((*x_init.shape[:-1], self.control_dim)).requires_grad_(True)
        optimizer = torch.optim.SGD([u_init], lr=0.01)
        for _ in range(100):
            xu = torch.cat((x_init, u_init), dim=-1)
            optimizer.zero_grad()
            # loss = xu.T @ Q @ xu + p @ xu
            loss = torch.einsum('xay,xyz,xza->xa', xu.unsqueeze(1), Q, xu.unsqueeze(2)).squeeze(-1)\
                  + torch.einsum('xay,xya->xa', p.unsqueeze(1), xu.unsqueeze(2)).squeeze(-1)
            loss = torch.sum(loss)
            loss.backward(retain_graph=True)
            optimizer.step()
        return u_init # return optimized control input

        # F = torch.cat((self.A, self.B), dim=1).unsqueeze(0).unsqueeze(0).repeat(self.T, num_batches, 1, 1)
        # x_pred, u_pred, objs_pred = MPC(
        #     self.state_dim, self.control_dim, self.T,
        #     u_lower=None, u_upper=None, u_init=None,
        #     lqr_iter=100,
        #     verbose=-1,
        #     exit_unconverged=False,
        #     detach_unconverged=False,
        #     n_batch=num_batches,
        # )(x_init, QuadCost(Q, p), LinDx(F))
        # return u




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
            seed=0,
            n_steps=256, # 2048 rollout len by default, but this takes too long for MPC with LQR solver
            n_epochs=10,
            learning_rate=3e-4
        )
        print("Created ACMPC model")
    
    def train(self):
        fp = "mpc_output_nowind.json"
        data = []
        for i in range(1000):
            self.actor_critic.learn(total_timesteps=32, log_interval=1)
            self.actor_critic.save("quadplus_mpc_nowind")
            obs = self.actor_critic.env.reset()
            
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