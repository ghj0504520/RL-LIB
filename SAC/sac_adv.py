import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib.pyplot as plt
from collections import namedtuple

import argparse
import time


# Target copy utility
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# CUDA GPU device usage utility
deviceGPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data):
    """Move a tensor or a collection of tensors to the specified device."""
    if isinstance(data, (list, tuple)):
        return [to_device(d, deviceGPU) for d in data]
    return data.to(deviceGPU)


# Replay buffer utility
Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) #state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.memory)


# GYM ENV ActionWrapper for Normalization
class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, -1, 1)
        
        return action


# SAC Q net
class SoftQNetwork(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_actions,  init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights for the final layer
        self.layers[-1].weight.data.uniform_(-init_w, init_w)
        self.layers[-1].bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = self.layers(x)
        return x
        

# SAC policy net
class PolicyNetwork(nn.Module):
    def __init__(self,hidden_size, num_inputs, num_actions, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.shared_layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),  # Replace with the provided activation if needed
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = self.shared_layers(state)
        mean    = (self.mean_linear(x))

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape) 
        action_0 = torch.tanh(mean + std*z.to(deviceGPU)) # TanhNormal distribution as actions; reparameterization trick
        action = action_0 #self.action_range*action_0

        '''
         The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
         The TanhNorm forces the Gaussian with infinite action range to be finite. \
         For the three terms in this log-likelihood estimation: \
         (1). the first term is the log probability of action as in common \
         stochastic Gaussian action policy (without Tanh); \
         (2). the second term is the caused by the Tanh(), \
         as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
         the epsilon is for preventing the negative cases in log; \
        '''
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(deviceGPU)) - torch.log(1. - action_0.pow(2) + epsilon) #-  np.log(self.action_range)
        
        '''
         both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
         the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
         needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
         '''
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std


# Advanced SOFT ACTOR and CRITIC AGENT
class ADVSOFTACTORCRITICAGENT():
    def __init__(self, num_inputs, num_actions, hidden_dim, q_lr, p_lr, a_lr):
        super(ADVSOFTACTORCRITICAGENT, self).__init__()

        self.soft_q_net1 = SoftQNetwork(hidden_dim, num_inputs, num_actions).to(deviceGPU)
        self.soft_q_net2 = SoftQNetwork(hidden_dim, num_inputs, num_actions).to(deviceGPU)
        self.target_soft_q_net1 = SoftQNetwork(hidden_dim, num_inputs, num_actions).to(deviceGPU)
        self.target_soft_q_net2 = SoftQNetwork(hidden_dim, num_inputs, num_actions).to(deviceGPU)
        hard_update(self.target_soft_q_net1, self.soft_q_net1)
        hard_update(self.target_soft_q_net2, self.soft_q_net2)

        self.policy_net = PolicyNetwork(hidden_dim, num_inputs, num_actions).to(deviceGPU)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=deviceGPU)

        #print('Soft Q Network (1,2): ', self.soft_q_net1)
        #print('Policy Network: ', self.policy_net)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.num_actions = num_actions

    def action_selection(self, state=None):
        if state is None:
            a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
            return a.numpy()
       
        mean, log_std = self.policy_net(state)
        std = log_std.exp()
    
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape).to(deviceGPU)
        action = torch.tanh(mean + std*z) #self.action_range* torch.tanh(mean + std*z)
        action = action.detach().cpu().numpy()[0]
    
        return action        

    def update(self, batch, reward_scale=10., auto_alpha=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2):
        
        state = to_device(torch.cat([b.state for b in batch]))
        action = to_device(torch.cat([b.action for b in batch]))
        reward = to_device(torch.cat([b.reward for b in batch]).unsqueeze(1))
        done = to_device(torch.cat([b.mask for b in batch]).unsqueeze(1))
        next_state = to_device(torch.cat([b.next_state for b in batch]))

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)

        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; 
                                                                                           # plus a small number to prevent numerical problem
    
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_alpha is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        target_value_from_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_value_from_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  


        # Training Policy Function
        # reparameterization action, q value dependent on policy, cannot detach
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        soft_update(self.target_soft_q_net1, self.soft_q_net1, soft_tau)
        soft_update(self.target_soft_q_net2, self.soft_q_net2, soft_tau)
        return policy_loss, q_value_loss1+q_value_loss2




# MAIN
# choose env
#ENV = ['Pendulum-v1'][1]

env = NormalizedActions(gym.make("Pendulum-v1"))
action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]

env.reset(seed=1)
np.random.seed(1)
random.seed(1)
torch.manual_seed(1) # config for CPU
torch.cuda.manual_seed(1) # config for GPU

# hyper-parameters for RL training
hidden_dim = 512
soft_q_lr = 3e-4
policy_lr = 3e-4
alpha_lr  = 3e-4

replay_buffer_size = 1e6
target_entropy = -1.*action_dim
max_episodes  = 500
max_steps = 200  # Pendulum needs 150 steps per episode to learn well
frame_idx   = 0
batch_size  = 300
explore_steps = 0  # for random action sampling in the beginning of training
reward_scale=10.
AUTO_ALPHA=True

q_losses = []
p_losses = []
all_rewards = []
ewma_reward = 0

replay_buffer = ReplayMemory(replay_buffer_size)
advSACAgent=ADVSOFTACTORCRITICAGENT(state_dim, action_dim, hidden_dim, soft_q_lr, policy_lr, alpha_lr)

if __name__ == '__main__':
        
    # training loop
    for episode_idx in range(max_episodes):
        initState, _ = env.reset()
        state = torch.Tensor([initState]).to(deviceGPU)
        
        episode_reward = 0.0
        episode_q_loss = 0.0
        episode_p_loss = 0.0
        step = 0
        
        terminate = False
        truncated = False   

        while not terminate and not truncated:
            step = step+1

            if frame_idx > explore_steps:
                action = advSACAgent.action_selection(state)
            else:
                action = advSACAgent.action_selection()
            
            next_state, reward, terminate, truncated, _ = env.step(action)
                
            replay_buffer.push(
                state, 
                torch.tensor([action], device=deviceGPU), 
                torch.tensor([terminate], dtype=torch.float32, device=deviceGPU),
                torch.tensor([next_state], device=deviceGPU),
                torch.tensor([reward], dtype=torch.float32, device=deviceGPU)
                )
            
            state = torch.tensor([next_state], device=deviceGPU)
            episode_reward += reward
            frame_idx += 1
            
            
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                ploss, qloss=advSACAgent.update(batch, reward_scale, auto_alpha=AUTO_ALPHA, target_entropy=target_entropy)

                episode_p_loss = episode_p_loss+ploss.item()
                episode_q_loss = episode_q_loss + qloss.item()*0.5

            if terminate:
                break

        q_losses.append(episode_q_loss/step)
        p_losses.append(episode_p_loss/step)
        ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
        all_rewards.append(ewma_reward)

        if episode_idx % 1 == 0:
            print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}\t p_loss: {}, q_loss: {}'.format(episode_idx, step, 
                                                                episode_reward, ewma_reward, episode_p_loss/step, episode_q_loss/step))

    # Plot the losses and EWMA reward after training
    plt.figure(figsize=(18, 6))
    
    # Plot policy loss
    plt.subplot(1, 3, 1)
    plt.plot(p_losses, label='Policy Loss')
    plt.title('Policy Loss Over Episodes')
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot value loss
    plt.subplot(1, 3, 2)
    plt.plot(q_losses, label='Value Loss', color='orange')
    plt.title('Value Loss Over Episodes')
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot EWMA reward
    plt.subplot(1, 3, 3)
    plt.plot(all_rewards, label='EWMA Reward', color='green')
    plt.title('EWMA Reward Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('EWMA Reward')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save the plot as an image
    plt.savefig('ADV_SAC_plot-whole.png')
    plt.show()

    advSACAgent.policy_net.eval()
    for eps in range(10):
        initState, _ = env.reset()
        state = torch.Tensor([initState]).to(deviceGPU)
        episode_reward = 0

        terminate = False
        truncated = False   

        while not terminate and not truncated:
            action = advSACAgent.action_selection(state)
            next_state, reward, terminate, truncated, _ = env.step(action)
            #env.render()   


            episode_reward += reward
            state = torch.tensor([next_state], device=deviceGPU)

        print('Test Episode: ', eps, '| Episode Reward: ', episode_reward)
