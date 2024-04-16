from model import NeuralCostMap


def mpc_norm():
    # guassian sample form diffmpc outptu
    pass

def get_sample_trajectories():
    # call sim to get sample trajectories
    pass

def train():
    policy_net = NeuralCostMap() # actor, cost map
    value_net = None # critic
    diff_mpc = None

    for i in range(500): # sample trajectories
        traj_samples = get_sample_trajectories()
        for traj in traj_samples:
            traj = mpc_norm(diff_mpc())
            # compute reward for traj
        rewards = None
        advantages = value_net(rewards)
        # policy grad (PPO clip) 
        # diff mpc update
        # fit cost map using regression based on MSE

