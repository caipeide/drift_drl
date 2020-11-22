import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.autograd import grad
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter


#CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'


parser = argparse.ArgumentParser()

parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--gradient_steps', default=1, type=int)

parser.add_argument('--learning_rate', default=3e-4, type=int)
parser.add_argument('--gamma', default=0.99, type=int) # discount gamma
parser.add_argument('--capacity', default=400000, type=int) # replay buffer size
parser.add_argument('--iteration', default=100000, type=int) #  num of  games
parser.add_argument('--batch_size', default=512, type=int) # mini batch size
parser.add_argument('--seed', default=1, type=int)

# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--num_hidden_units_per_layer', default=256, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=2000, type=int) #
parser.add_argument('--load', default=True, type=bool) # load model

args = parser.parse_args()

min_Val = torch.tensor(1e-7).float().to(device)

class Replay_buffer():
    def __init__(self, capacity,state_dim,action_dim):
        self.capacity = capacity
        self.state_pool = torch.zeros(self.capacity, state_dim).float().to(device)
        self.action_pool = torch.zeros(self.capacity, action_dim).float().to(device)
        self.reward_pool = torch.zeros(self.capacity, 1).float().to(device)
        self.next_state_pool = torch.zeros(self.capacity, state_dim).float().to(device)
        self.done_pool = torch.zeros(self.capacity, 1).float().to(device)
        self.num_transition = 0

    def push(self, s, a, r, s_, d):
        index = self.num_transition % self.capacity
        s = torch.tensor(s).float().to(device)
        a = torch.tensor(a).float().to(device)
        r = torch.tensor(r).float().to(device)
        s_ = torch.tensor(s_).float().to(device)
        d = torch.tensor(d).float().to(device)
        for pool, ele in zip([self.state_pool, self.action_pool, self.reward_pool, self.next_state_pool, self.done_pool],
                           [s, a, r, s_, d]):
            pool[index] = ele
        self.num_transition += 1

    def sample(self, batch_size):
        index = np.random.choice(range(self.capacity), batch_size, replace=False)
        bn_s, bn_a, bn_r, bn_s_, bn_d = self.state_pool[index], self.action_pool[index], self.reward_pool[index],\
                                        self.next_state_pool[index], self.done_pool[index]

        return bn_s, bn_a, bn_r, bn_s_, bn_d

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim ,min_log_std=-20, max_log_std=2):##max and min left to modify
        super(Actor, self).__init__()
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

        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std) ##give a resitriction on the chosen action
        return mu, log_std_head


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1) # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SACAgent():
    def __init__(self, state_dim = 45, action_dim=21):
        super(SACAgent, self).__init__()

        self.policy_net = Actor(state_dim=state_dim, action_dim = action_dim).to(device)
        self.value_net = Critic(state_dim).to(device)
        self.Target_value_net = Critic(state_dim).to(device)
        
        self.Q_net1 = Q(state_dim, action_dim).to(device)
        self.Q_net2 = Q(state_dim, action_dim).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.learning_rate)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=args.learning_rate)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=args.learning_rate)

        self.replay_buffer = Replay_buffer(args.capacity,state_dim,action_dim)
        self.num_transition = 0
        self.num_training = 0
        self.writer = SummaryWriter('./exp-SAC_dual_Q_network')

        self.value_criterion = nn.MSELoss()
        self.Q1_criterion = nn.MSELoss()
        self.Q2_criterion = nn.MSELoss()

        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.steer_range = (-0.8,0.8)
        self.throttle_range = (0.6,1.0)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        
        dist = Normal(mu, sigma)
        z = dist.sample()

        steer = float(torch.tanh(z[0,0]).detach().cpu().numpy())
        throttle = float(torch.tanh(z[0,1]).detach().cpu().numpy())
        
        steer = (steer + 1)/2 * (self.steer_range[1] - self.steer_range[0]) + self.steer_range[0]
        throttle = (throttle + 1)/2 * (self.throttle_range[1] - self.throttle_range[0]) + self.throttle_range[0]
        

        return np.array([steer, throttle])


    def test(self, state):
        state = torch.FloatTensor(state).to(device)
        mu, log_sigma = self.policy_net(state)
       
        action = mu

        steer = float(torch.tanh(action[0,0]).detach().cpu().numpy())
        throttle = float(torch.tanh(action[0,1]).detach().cpu().numpy())

        steer = (steer + 1)/2 * (self.steer_range[1] - self.steer_range[0]) + self.steer_range[0]
        throttle = (throttle + 1)/2 * (self.throttle_range[1] - self.throttle_range[0]) + self.throttle_range[0]


        return np.array([steer, throttle])

    def evaluate(self, state):
        batch = state.size()[0]
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        noise = Normal(0, 1)

        z = noise.sample()

        action = torch.tanh(batch_mu + batch_sigma * z.to(device))
        
        log_prob = dist.log_prob(batch_mu + batch_sigma * z.to(device)) - torch.log(1 - action.pow(2) + min_Val)

        log_prob_0 = log_prob[:,0].reshape(batch,1)
        log_prob_1 = log_prob[:,1].reshape(batch,1)
        log_prob = log_prob_0 + log_prob_1

        return action, log_prob, z, batch_mu, batch_log_sigma


    def update(self):
        if self.num_training % 500 == 0:
            print("**************************Train Start************************")
            print("Training ... \t{} times ".format(self.num_training))

        for _ in range(args.gradient_steps):
            bn_s, bn_a, bn_r, bn_s_, bn_d = self.replay_buffer.sample(args.batch_size)
            
            target_value = self.Target_value_net(bn_s_)
            next_q_value = bn_r + (1 - bn_d) * args.gamma * target_value

            excepted_value = self.value_net(bn_s)
            excepted_Q1 = self.Q_net1(bn_s, bn_a)
            excepted_Q2 = self.Q_net2(bn_s, bn_a)
            sample_action, log_prob, z, batch_mu, batch_log_sigma = self.evaluate(bn_s)
            excepted_new_Q = torch.min(self.Q_net1(bn_s, sample_action), self.Q_net2(bn_s, sample_action))
            next_value = excepted_new_Q - log_prob
            
            V_loss = self.value_criterion(excepted_value, next_value.detach()).mean()  # J_V

            # Dual Q net
            Q1_loss = self.Q1_criterion(excepted_Q1, next_q_value.detach()).mean() # J_Q
            Q2_loss = self.Q2_criterion(excepted_Q2, next_q_value.detach()).mean()

            pi_loss = (log_prob - excepted_new_Q).mean() # according to original paper

            self.writer.add_scalar('Loss/V_loss', V_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q1_loss', Q1_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q2_loss', Q2_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/policy_loss', pi_loss, global_step=self.num_training)

            # mini batch gradient descent
            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.Q1_optimizer.zero_grad()
            Q1_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            self.Q1_optimizer.step()

            self.Q2_optimizer.zero_grad()
            Q2_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            self.Q2_optimizer.step()

            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            # update target v net update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - args.tau) + param * args.tau)

            self.num_training += 1

    def save(self,epoch, capacity):
        os.makedirs('./SAC_model_' +str(capacity) , exist_ok=True)
        torch.save(self.policy_net.state_dict(), './SAC_model_' +str(capacity)+ '/policy_net_' + str(epoch) + '.pth')
        torch.save(self.value_net.state_dict(), './SAC_model_'  +str(capacity)+ '/value_net_'+ str(epoch) +'.pth')
        torch.save(self.Q_net1.state_dict(), './SAC_model_' +str(capacity)+'/Q_net1_' + str(epoch) + '.pth')
        torch.save(self.Q_net2.state_dict(), './SAC_model_' +str(capacity)+'/Q_net2_' + str(epoch) + '.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, epoch, capacity):
        dir = './SAC_model_' + str(capacity) + '/'
        self.policy_net.load_state_dict(torch.load( dir + 'policy_net_' + str(epoch) + '.pth'))
        self.value_net.load_state_dict(torch.load( dir + 'value_net_'+ str(epoch) + '.pth'))
        self.Q_net1.load_state_dict(torch.load( dir + 'Q_net1_' + str(epoch) + '.pth'))
        self.Q_net2.load_state_dict(torch.load( dir + 'Q_net2_' + str(epoch) + '.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")