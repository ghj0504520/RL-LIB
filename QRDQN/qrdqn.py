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
        self.memory[self.position] = Transition( *args)
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
class QRCritic(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs, num_supports):
        super(QRCritic, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.num_supports = num_supports

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.num_outputs * self.num_supports)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        theta = x.view(-1, self.num_outputs, self.num_supports)
    
        return theta
    
    def get_action(self, state):
        theta = self.forward(state)
        Q = theta.mean(dim=2, keepdim=True)
        action = torch.argmax(Q)
        return action.item()

# QRDQN AGENT
class QRDQNAGENT(object):
    def __init__(self, num_inputs, num_actions, num_supports):
        super(QRDQNAGENT, self).__init__()

        self.qrcritic = QRCritic(128, num_inputs, num_actions, num_supports).to(deviceGPU)
        self.qrcritic_target = QRCritic(128, num_inputs, num_actions, num_supports).to(deviceGPU)
        hard_update(self.qrcritic_target, self.qrcritic)

        self.optim = optim.Adam(self.qrcritic.parameters(), lr=lr)
    
        self.num_actions = num_actions
        self.num_supports = num_supports

    def action_selection(self, state, epsilon, env):
        if np.random.rand() <= epsilon:
            return env.action_space.sample()
        else:
            return self.qrcritic.get_action(state)

    def update_parameter(self, batch):

        states = to_device(torch.cat([b.state for b in batch]).unsqueeze(1))
        actions = to_device(torch.cat([b.action for b in batch]))
        rewards = to_device(torch.cat([b.reward for b in batch]))
        masks = to_device(torch.cat([b.mask for b in batch]))
        next_states = to_device(torch.cat([b.next_state for b in batch]).unsqueeze(1))

        theta = self.qrcritic(states)
        action = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, self.num_supports)
        theta_a = theta.gather(1, action).squeeze(1)

        next_theta = self.qrcritic_target(next_states) # batch_size * action * num_support
        next_action = next_theta.mean(dim=2).max(1)[1] # batch_size
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.num_supports)
        next_theta_a = next_theta.gather(1, next_action).squeeze(1) # batch_size * num_support

        T_theta = rewards.unsqueeze(1) + gamma * next_theta_a * (1-masks.unsqueeze(1))

        T_theta_tile = T_theta.view(-1, self.num_supports, 1).expand(-1, self.num_supports, self.num_supports)
        theta_a_tile = theta_a.view(-1, 1, self.num_supports).expand(-1, self.num_supports, self.num_supports)
        
        error_loss = T_theta_tile - theta_a_tile            
        huber_loss = F.smooth_l1_loss(theta_a_tile, T_theta_tile.detach(), reduction='none')
        tau = torch.arange(0.5 * (1 / self.num_supports), 1, 1 / self.num_supports).to(deviceGPU).view(1, self.num_supports)
        
        loss = (tau - (error_loss < 0).float()).abs() * huber_loss
        loss = loss.mean(dim=2).sum(dim=1).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss

env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 64
lr = 0.001
initial_exploration = 1000
goal_score = 480
log_interval = 10
update_target = 100
replay_memory_capacity = 10000

num_support = 8

def main():
    env = gym.make(env_name)
    rSeed = 9
    env.action_space.seed(rSeed)
    env.reset(seed=rSeed)
    np.random.seed(rSeed)
    random.seed(rSeed)
    torch.manual_seed(rSeed) # config for CPU
    torch.cuda.manual_seed(rSeed) # config for GPU

    qrdqnAgent = QRDQNAGENT(env.observation_space.shape[0], env.action_space.n, num_support)
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
            action = qrdqnAgent.action_selection(state, epsilon, env)
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
                tmpLoss = qrdqnAgent.update_parameter(training_batch)
                loss += tmpLoss.item()
                if steps % update_target == 0:
                    hard_update(qrdqnAgent.qrcritic_target, qrdqnAgent.qrcritic)
        
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
    plt.savefig('QRDQN_plot-whole.png')
    plt.show()


if __name__=="__main__":
    main()
