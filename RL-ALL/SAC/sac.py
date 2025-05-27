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
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
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


# SAC value net
class ValueNetwork(nn.Module):
    def __init__(self, hidden_size, num_inputs, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights for the final layer
        self.layers[-1].weight.data.uniform_(-init_w, init_w)
        self.layers[-1].bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = self.layers(state)
        return x
        
# SAC q net
class SoftQNetwork(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_actions, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_size),
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
    def __init__(self, hidden_size, num_inputs, num_actions, init_w=3e-3, log_std_min=-20, log_std_max=2):
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

        ''' stochastic evaluation '''
        log_prob = Normal(mean, std).log_prob(mean + std*z.to(deviceGPU)) - torch.log(1. - action_0.pow(2) + epsilon) #-  np.log(self.action_range)
        ''' deterministic evaluation '''
        # log_prob = Normal(mean, std).log_prob(mean) - torch.log(1. - torch.tanh(mean).pow(2) + epsilon) #-  np.log(self.action_range)

        '''
         both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
         the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
         needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
         '''
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std


# SOFT ACTOR and CRITIC AGENT
class SOFTACTORCRITICAGENT(object):
    def __init__(self, num_inputs, num_actions, hidden_dim, v_lr, q_lr, p_lr):
        super(SOFTACTORCRITICAGENT, self).__init__()

        self.value_net        = ValueNetwork(hidden_dim, num_inputs).to(deviceGPU)
        self.target_value_net = ValueNetwork(hidden_dim, num_inputs).to(deviceGPU)
        hard_update(self.target_value_net, self.value_net)

        self.soft_q_net1 = SoftQNetwork(hidden_dim, num_inputs, num_actions).to(deviceGPU)
        self.soft_q_net2 = SoftQNetwork(hidden_dim, num_inputs, num_actions).to(deviceGPU)
        self.policy_net = PolicyNetwork(hidden_dim, num_inputs, num_actions).to(deviceGPU)

        #print('(Target) Value Network: ', self.value_net)
        #print('Soft Q Network (1,2): ', self.soft_q_net1)
        #print('Policy Network: ', self.policy_net)

        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=v_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=p_lr)
    
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

    def update_parameter(self, batch, reward_scale, gamma=0.99,soft_tau=1e-2):
        alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q)
                
        state = to_device(torch.cat([b.state for b in batch]))
        action = to_device(torch.cat([b.action for b in batch]))
        reward = to_device(torch.cat([b.reward for b in batch]).unsqueeze(1))
        done = to_device(torch.cat([b.mask for b in batch]).unsqueeze(1))
        next_state = to_device(torch.cat([b.next_state for b in batch]))
        
        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_value    = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        reward = reward_scale*(reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std

        # Training Q Function
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * gamma * target_value # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  


        # Training Value Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),self.soft_q_net2(state, new_action))
        target_value_func = predicted_new_q_value - alpha * log_prob # for stochastic training, it equals to expectation over action
        value_loss = self.value_criterion(predicted_value, target_value_func.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


        # Training Policy Function
        # reparameterization action, q value dependent on policy, cannot detach
        policy_loss = (alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # print('value_loss: ', value_loss)
        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        soft_update(self.target_value_net, self.value_net, soft_tau)
        return policy_loss, value_loss




# MAIN
# choose env
env = NormalizedActions(gym.make("Pendulum-v1"))
action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]

env.reset(seed=1)
np.random.seed(1)
random.seed(1)
torch.manual_seed(1) # config for CPU
torch.cuda.manual_seed(1) # config for GPU

# hyper-parameters
hidden_dim = 512
value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

replay_buffer_size = int(1e6)

max_episodes  = 500
max_steps   = 200  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
frame_idx   = 0
batch_size  = 128
explore_steps = 0
reward_scale=10.0

v_losses = []
p_losses = []
all_rewards = []
ewma_reward = 0

replay_buffer = ReplayMemory(replay_buffer_size)
sacAgent = SOFTACTORCRITICAGENT(state_dim, action_dim, hidden_dim, value_lr, soft_q_lr, policy_lr)

if __name__ == '__main__':

    # training loop
    for episode_idx in range(max_episodes):
        initState, _ = env.reset()
        state = torch.Tensor([initState]).to(deviceGPU)
        
        episode_reward = 0.0
        episode_v_loss = 0.0
        episode_p_loss = 0.0
        step = 0
        
        terminate = False
        truncated = False   

        while not terminate and not truncated:
            step = step+1

            if frame_idx >= explore_steps:
                action = sacAgent.action_selection(state)
            else:
                action = sacAgent.action_selection()
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
                ploss, vloss=sacAgent.update_parameter(batch, reward_scale)
            
                episode_p_loss = episode_p_loss+ploss.item()
                episode_v_loss = episode_v_loss+vloss.item()

            if terminate:
                break

        v_losses.append(episode_v_loss/step)
        p_losses.append(episode_p_loss/step)
        ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
        all_rewards.append(ewma_reward)

        if episode_idx % 1 == 0:
            print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}\t p_loss: {}, v_loss: {}'.format(episode_idx, step, 
                                                                episode_reward, ewma_reward, episode_p_loss/step, episode_v_loss/step))

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
    plt.plot(v_losses, label='Value Loss', color='orange')
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
    plt.savefig('SAC_plot-whole.png')
    plt.show()


    sacAgent.policy_net.eval()
    for eps in range(10):
        initState, _ = env.reset()
        state = torch.Tensor([initState]).to(deviceGPU)
        episode_reward = 0

        terminate = False
        truncated = False   

        while not terminate and not truncated:
            action = sacAgent.action_selection(state)
            next_state, reward, terminate, truncated, _ = env.step(action)

            episode_reward += reward
            state = torch.tensor([next_state], device=deviceGPU)

        print('Test Episode: ', eps, '| Episode Reward: ', episode_reward)
