import matplotlib.pyplot as plt
import json
import numpy as np

f = open('ppo_output_nowind.json')
data = json.load(f)
x = np.cumsum([d['time/total_timesteps'] for d in data])
y = [d['rollout/ep_rew_mean'] for d in data]
plt.plot(x,y)
plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Reward")
plt.title("PPO for Drone Hover Objective (No Wind)")
f.close()
plt.show()