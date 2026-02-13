import os
import sys
import gym
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from collections import namedtuple
import matplotlib.pyplot as plt

# Target copy utility
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# Replay buffer utility
Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

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

# NN model
class ImplicitCritic(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs, quantile_embedding_dim):
        super(ImplicitCritic, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.quantile_embedding_dim = quantile_embedding_dim

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_outputs)
        self.phi = nn.Linear(self.quantile_embedding_dim, hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, state, tau, num_quantiles):
        input_size = state.size()[0] # batch_size(train) or 1(get_action)
        tau = tau.expand(input_size * num_quantiles, self.quantile_embedding_dim)
        pi_mtx = torch.Tensor(
            np.pi * np.arange(
                0, self.quantile_embedding_dim)).to(deviceGPU).expand(input_size * num_quantiles, self.quantile_embedding_dim)
        cos_tau = torch.cos(tau * pi_mtx)

        phi = self.phi(cos_tau)
        phi = F.relu(phi)

        state_tile = state.expand(input_size, num_quantiles, self.num_inputs)
        state_tile = state_tile.flatten().view(-1, self.num_inputs)
        
        x = F.relu(self.fc1(state_tile))
        x = self.fc2(x * phi)
        z = x.view(-1, num_quantiles, self.num_outputs)

        z = z.transpose(1, 2) # [input_size, num_output, num_quantile]
        return z

    def get_action(self, state):
        tau = torch.Tensor(np.random.rand(num_quantile_sample, 1) * 0.5).to(deviceGPU) # CVaR
        z = self.forward(state, tau, num_quantile_sample)
        q = z.mean(dim=2, keepdim=True)
        action = torch.argmax(q)
        return action.item()

# IQN AGENT
class IQNAGENT(object):
    def __init__(self, num_inputs, num_actions, quantile_embedding_dim):
        super(IQNAGENT, self).__init__() 

        self.icritic = ImplicitCritic(128, num_inputs, num_actions, quantile_embedding_dim).to(deviceGPU)
        self.icritic_target = ImplicitCritic(128, num_inputs, num_actions, quantile_embedding_dim).to(deviceGPU)
        hard_update(self.icritic_target, self.icritic)

        self.optim = optim.Adam(self.icritic.parameters(), lr=lr)
    
        self.num_actions = num_actions

    def action_selection(self, state, epsilon, env):
        if np.random.rand() <= epsilon:
            return env.action_space.sample()
        else:
            return self.icritic.get_action(state)

    def update_parameter(self, batch):
        states = to_device(torch.cat([b.state for b in batch]).unsqueeze(1))
        actions = to_device(torch.cat([b.action for b in batch]))
        rewards = to_device(torch.cat([b.reward for b in batch]))
        masks = to_device(torch.cat([b.mask for b in batch]))
        next_states = to_device(torch.cat([b.next_state for b in batch]).unsqueeze(1))

        # prediction
        tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1)).to(deviceGPU)
        z = self.icritic(states, tau, num_tau_sample)
        action = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, num_tau_sample)
        z_a = z.gather(1, action).squeeze(1)

        # td target
        tau_prime = torch.Tensor(np.random.rand(batch_size * num_tau_prime_sample, 1)).to(deviceGPU)
        next_z = self.icritic_target(next_states, tau_prime, num_tau_prime_sample)
        next_action = next_z.mean(dim=2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_tau_prime_sample)
        next_z_a = next_z.gather(1, next_action).squeeze(1)

        T_z = rewards.unsqueeze(1) + gamma * next_z_a * (1-masks.unsqueeze(1))

        T_z_tile = T_z.view(-1, num_tau_prime_sample, 1).expand(-1, num_tau_prime_sample, num_tau_sample)
        z_a_tile = z_a.view(-1, 1, num_tau_sample).expand(-1, num_tau_prime_sample, num_tau_sample)
        
        error_loss = T_z_tile - z_a_tile
        huber_loss = F.smooth_l1_loss(z_a_tile, T_z_tile.detach(), reduction='none')
        tau = torch.arange(0, 1, 1 / num_tau_sample).to(deviceGPU).view(1, num_tau_sample)
        
        loss = (tau - (error_loss < 0).float()).abs() * huber_loss
        loss = loss.mean(dim=2).sum(dim=1).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss

env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 1000
goal_score = 463
log_interval = 10
update_target = 100
replay_memory_capacity = 1000

num_quantile_sample = 32
num_tau_sample = 16
num_tau_prime_sample = 8
quantile_embedding_dim = 64

def main():
    env = gym.make(env_name)
    rSeed = 7
    env.action_space.seed(rSeed)
    env.reset(seed=rSeed)
    np.random.seed(rSeed)
    random.seed(rSeed)
    torch.manual_seed(rSeed) # config for CPU
    torch.cuda.manual_seed(rSeed) # config for GPU

    iqnAgent = IQNAGENT(env.observation_space.shape[0], env.action_space.n, quantile_embedding_dim)
    replay_buffer = ReplayMemory(replay_memory_capacity)
    ewma_reward = 0
    loss = 0
    losses = []
    all_rewards = []

    epsilon = 1.0
    steps = 0

    for episode_idx in range(10000):
        initState,_ = env.reset()
        state = torch.Tensor(initState).to(deviceGPU)
        state = state.unsqueeze(0)

        terminate = False
        truncated = False
        episode_reward = 0
        loss = 0
        step = 0
        while not terminate and not truncated:
            steps += 1
            step += 1
            action = iqnAgent.action_selection(state, epsilon, env)
            next_state, reward, terminate, truncated, _ = env.step(action)

            next_state = torch.Tensor(next_state).to(deviceGPU)
            next_state = next_state.unsqueeze(0)

            replay_buffer.push(
                state, 
                torch.tensor([action], device=deviceGPU), 
                torch.tensor([terminate], dtype=torch.float32, device=deviceGPU), 
                next_state, 
                torch.tensor([reward], device=deviceGPU)
            )
            episode_reward += reward
            state = next_state

            if steps > initial_exploration:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                training_batch = replay_buffer.sample(batch_size)
                tmpLoss = iqnAgent.update_parameter(training_batch)
                loss += tmpLoss.item()
                if steps % update_target == 0:
                    hard_update(iqnAgent.icritic_target, iqnAgent.icritic)

        losses.append(loss/step)

        ewma_reward =  (1 - 0.05) * ewma_reward + 0.05 * episode_reward
        
        all_rewards.append(ewma_reward)

        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}\t loss: {}'.format(episode_idx, step, 
                                                                                       episode_reward, ewma_reward, loss/step))
        # early terminate
        if ewma_reward > goal_score:
            break
    

    chunk_size = 1  # Every 1 episodes
    num_chunks = len(all_rewards) // chunk_size  # Number of chunks


    # Plotting utility
    # Calculate average reward for every 200 episodes
    average_rewards = [sum(all_rewards[i * chunk_size:(i + 1) * chunk_size]) / chunk_size for i in range(num_chunks)]
    average_loss = [sum(losses[i * chunk_size:(i + 1) * chunk_size]) / chunk_size for i in range(num_chunks)]

    # Plot the average reward and loss every 200 episodes
    plt.figure(figsize=(12, 6))

    # Plot for reward
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_chunks + 1), average_rewards, label="Average Reward")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Average Reward Every Episodes')
    plt.grid(True)

    # Plot for loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_chunks + 1), average_loss, label="Average Loss", color='r')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Average Loss Every Episodes')
    plt.grid(True)

    plt.tight_layout()

    # Show legend
    plt.legend()

    # Save the plot as an image
    plt.savefig('IQN_plot-whole.png')
    plt.show()

if __name__=="__main__":
    main()
