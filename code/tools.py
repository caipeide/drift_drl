import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
steer_range = (-0.8,0.8)
throttle_range = (0.6,1.0)

def getHeading(env):
	transform = env.world.player.get_transform()
	ego_yaw = transform.rotation.yaw
	if ego_yaw < 0:
		ego_yaw += 360
	if ego_yaw > 360:
		ego_yaw -= 360
	if ego_yaw > 180:
		ego_yaw = -(360-ego_yaw)
	return ego_yaw

def bool2num(flag):
	if flag == True:
		return 1
	else:
		return 0

class SAC_Actor(nn.Module):
	# reference : https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/tree/master/Char09%20SAC
	def __init__(self, state_dim, action_dim ,min_log_std=-20, max_log_std=2):
		super(SAC_Actor, self).__init__()
		self.fc1 = nn.Linear(state_dim, 512)
		self.fc2 = nn.Linear(512, 256)
		self.mu_head = nn.Linear(256, action_dim)
		self.log_std_head = nn.Linear(256, action_dim)

		self.min_log_std = min_log_std
		self.max_log_std = max_log_std

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		mu = self.mu_head(x)
		log_std_head = self.log_std_head(x)

		log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
		
		return mu, log_std_head

	def test(self, state):
		state = torch.FloatTensor(state).to(device)
		mu, log_sigma = self(state)
		action = mu
		steer = float(torch.tanh(action[0,0]).detach().cpu().numpy())
		throttle = float(torch.tanh(action[0,1]).detach().cpu().numpy())

		steer = (steer + 1)/2 * (steer_range[1] - steer_range[0]) + steer_range[0]
		throttle = (throttle + 1)/2 * (throttle_range[1] - throttle_range[0]) + throttle_range[0]

		return np.array([steer, throttle])

class DDPG_Actor(nn.Module):
	# reference: https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/tree/master/Char05%20DDPG
    def __init__(self, state_dim, action_dim):
        super(DDPG_Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x

    def test(self, state):
        state = torch.FloatTensor(state).to(device)
        z = self(state)
        steer = float(z[0,0].detach().cpu().numpy())
        throttle = float(z[0,1].detach().cpu().numpy())

        steer = (steer + 1)/2 * (steer_range[1] - steer_range[0]) + steer_range[0]
        throttle = (throttle + 1)/2 * (throttle_range[1] - throttle_range[0]) + throttle_range[0]

        return np.array([steer, throttle])