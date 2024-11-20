import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import matplotlib.pyplot as plt
from collections import namedtuple

import sys
import os
import time


# Target copy utility
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# Replay buffer utility
Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):

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

# CUDA GPU device usage utility
deviceGPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data):
    """Move a tensor or a collection of tensors to the specified device."""
    if isinstance(data, (list, tuple)):
        return [to_device(d, deviceGPU) for d in data]
    return data.to(deviceGPU)

# Ornstein-Uhlenbeck process for exploration
class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

## NN model for Actor
class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Actor, self).__init__()
        # self.action_space = action_space
        # num_outputs = action_space.shape[0]

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )
        
    def forward(self, inputs):
        # action range [-1,1]
        return self.layers(inputs)
        

## NN model for Critic
class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs1, num_inputs2):
        super(Critic, self).__init__()
        # self.action_space = action_space
        # num_outputs = action_space.shape[0]
        
        # critic need both state and action as input
        self.layers = nn.Sequential(
            nn.Linear(num_inputs1+num_inputs2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, inputs, actions):        
        return self.layers(torch.cat([inputs, actions], dim=-1))
    

class DDPGAGENT(object):
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.0005, hidden_size=128, lr_a=1e-4, lr_c=1e-3,
                policy_noise=0.2, noise_clip=0.5, policy_delay=2):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space.shape[0]).to(deviceGPU)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space.shape[0]).to(deviceGPU)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_a)

        self.critic1 = Critic(hidden_size, self.num_inputs, self.action_space.shape[0]).to(deviceGPU)
        self.critic1_target = Critic(hidden_size, self.num_inputs, self.action_space.shape[0]).to(deviceGPU)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr_c)

        self.critic2 = Critic(hidden_size, self.num_inputs, self.action_space.shape[0]).to(deviceGPU)
        self.critic2_target = Critic(hidden_size, self.num_inputs, self.action_space.shape[0]).to(deviceGPU)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau
        # TD3
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0  # Counter for policy delay

        hard_update(self.actor_target, self.actor) 
        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)

    def select_action(self, state, action_noise=None):
        # figure out, state is already in device
        
        # if your model has Dropout or BatchNorm, needing to set this
        self.actor.eval()

        # Disable gradient calculations
        with torch.no_grad():
            # Get the action from the actor (policy network)
            mu_action = self.actor(state)*2

            # Add noise for exploration, if provided
            if action_noise is not None:
                action_noise = to_device(torch.FloatTensor(action_noise))
                mu_action = mu_action + action_noise # Add noise to the action

            # Clip the action to be within the valid action space
            action_min = to_device(torch.tensor(self.action_space.low, dtype=mu_action.dtype))
            action_max = to_device(torch.tensor(self.action_space.high, dtype=mu_action.dtype))
            action_prob = torch.clamp(mu_action, action_min, action_max) # apply clip on probability
        return action_prob

    def update_parameters(self, batch):
        self.total_it += 1
        state_batch = to_device(torch.cat([b.state for b in batch]))
        action_batch = to_device(torch.cat([b.action for b in batch]))
        reward_batch = to_device(torch.cat([b.reward for b in batch]))
        mask_batch = to_device(torch.cat([b.mask for b in batch]))
        next_state_batch = to_device(torch.cat([b.next_state for b in batch]))
        
        # Calculate policy loss and value loss
        # Update the actor and the critic

        # evaluate policy by target network instead of interaction network
        with torch.no_grad():
            next_action_batch = self.actor_target(next_state_batch)*2

            # TD3 trick one add noise when training
            action_noise = torch.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),
                                    dtype=torch.float32, device=deviceGPU)
            action_noise = torch.clamp(action_noise, -self.noise_clip, self.noise_clip)
            next_action_batch = torch.clamp(next_action_batch+action_noise, 
                                    torch.tensor(self.action_space.low, dtype=next_action_batch.dtype, device=deviceGPU),
                                    torch.tensor(self.action_space.high, dtype=next_action_batch.dtype, device=deviceGPU))

            # TD3 trick two similar to Double Q network
            next_q_values1 = self.critic1_target(next_state_batch, next_action_batch) # .detach()
            next_q_values2 = self.critic2_target(next_state_batch, next_action_batch)
            next_q_values = torch.min(next_q_values1, next_q_values2)
            # use target network to calculate target for evaluating loss
            target_q_values = reward_batch.view(-1,1) + (self.gamma * next_q_values * (1 - mask_batch.view(-1,1)))
        
        # update the both interaction critic network (with target network)
        # using unsqueeze(1) to avoid shape mismatching
        cur_q_values1 = self.critic1(state_batch, action_batch.unsqueeze(1))
        cur_q_values2 = self.critic2(state_batch, action_batch.unsqueeze(1))

        # update the interaction critic network
        value_loss1 = F.mse_loss(target_q_values, cur_q_values1) # .detach()
        self.critic1_optim.zero_grad()
        value_loss1.backward()
        self.critic1_optim.step()

        value_loss2 = F.mse_loss(target_q_values, cur_q_values2) # .detach()
        self.critic2_optim.zero_grad()
        value_loss2.backward()
        self.critic2_optim.step()

        policy_loss = None
        # TD3 trick three delayed policy update
        if self.total_it % self.policy_delay == 0:
            # use interaction critic network
            # only self.actor which be put in bracket will be tracked and calculated gradient
            policy_loss = -self.critic1(state_batch, self.actor(state_batch) * 2) # can use both critic1 or critic2
            
            # update the interaction actor network
            policy_loss = policy_loss.mean()
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
        
            # Soft update target networks
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic1_target, self.critic1, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)

        return value_loss1, policy_loss

def train():    
    num_episodes = 300
    gamma = 0.995
    tau = 0.002
    hidden_size = 256
    noise_scale = 0.3
    replay_size = 100000
    batch_size = 128
    updates_per_step = 1
    print_freq = 1
    ewma_reward = 0
    ewma_reward_history = []
    p_losses = []
    v_losses = []
    total_numsteps = 0
    updates = 0
    lr_a=3e-4
    lr_c=3e-3
    
    ddpgAgent = DDPGAGENT(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size,lr_a, lr_c)
    ounoise = OUNoise(env.action_space.shape[0])
    replay_buffer = ReplayMemory(replay_size)
    
    for episode_idx in range(num_episodes):
        
        ounoise.scale = noise_scale
        ounoise.reset()
        initState, info = env.reset()
        state = torch.Tensor([initState]).to(deviceGPU)
    
        terminate = False
        truncated = False
        episode_reward = 0
        step = 0
        episode_v_losses = 0
        episode_p_losses = 0
        while not terminate and not truncated:
            step = step +1
            
            action = ddpgAgent.select_action(state, ounoise.noise())
            next_state, reward, terminate, truncated, info = env.step(action.cpu().numpy()[0])

            done = terminate
            replay_buffer.push(
                state, 
                torch.tensor([action], device=deviceGPU), 
                torch.tensor([done], dtype=torch.float32, device=deviceGPU), 
                torch.tensor([next_state], device=deviceGPU), 
                torch.tensor([reward], dtype=torch.float32, device=deviceGPU)
            )

            state = torch.tensor([next_state], device=deviceGPU)  # Ensure next state is on the correct device
            
            episode_reward += reward
            if done:
                break

            if replay_buffer.__len__() >= batch_size and total_numsteps%updates_per_step == 0:
                training_batch = replay_buffer.sample(batch_size)
                value_loss, policy_loss = ddpgAgent.update_parameters(training_batch)

                if policy_loss != None:
                    episode_p_losses += policy_loss.item()
                episode_v_losses += value_loss.item()

        # update EWMA reward and log the results
        p_losses.append(episode_p_losses/step)
        v_losses.append(episode_v_losses/step)
        ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
        ewma_reward_history.append(ewma_reward)           
        
        if episode_idx % 1 == 0:
            print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}\t p_loss: {}\t v_loss: {}'.format(episode_idx, step, 
                                                                                       episode_reward, ewma_reward, 
                                                                                       episode_p_losses/step, episode_v_losses/step))
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
    plt.plot(ewma_reward_history, label='EWMA Reward', color='green')
    plt.title('EWMA Reward Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('EWMA Reward')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save the plot as an image
    plt.savefig('TD3_plot-whole.png')
    plt.show()

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    env = gym.make('Pendulum-v1')#,render_mode="human")
    #torch.manual_seed(random_seed)  
    #np.random.seed(random_seed) # figure out! fix the random seed
    #random.seed(random_seed) # figure out! fix the random seed
    train()
    #test(name='Pendulum-v1_0.0003_0.003_05042024_124142')

