import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
from collections import namedtuple
from collections import deque


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


# NN model
class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Critic, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )

    def forward(self, inputs):
        return self.layers(inputs)
    

# DQN with SOFT AGENT
class SOFTDQNAGENT(object):
    def __init__(self, num_inputs, num_actions):
        super(SOFTDQNAGENT, self).__init__()

        # num_inputs: env.observation_space.shape[0]
        # num_actions: env.action_space.n
        self.critic = Critic(256, num_inputs, num_actions).to(deviceGPU)
        self.critic_target = Critic(256, num_inputs, num_actions).to(deviceGPU)
        
        self.optim = optim.Adam(self.critic.parameters(), lr=1e-4)

        hard_update(self.critic_target, self.critic)
        self.learn_steps = 0

    # stable Soft V calculation so as to prevent overflow
    def getV(self, q_value): 
        # Find max Q-value for each state
        max_q = torch.max(q_value, dim=1, keepdim=True)[0]  # Shape: [batch_size, 1]
        
        # Compute stable exponentials
        exp_values = torch.exp((q_value - max_q) / ALPHA)  # Subtract max_q for stability
        
        # Compute sum of exponentials
        sum_exp = torch.sum(exp_values, dim=1, keepdim=True)  # Shape: [batch_size, 1]
        
        # Compute the soft value function using the log-sum-exp trick
        v = max_q + ALPHA * torch.log(sum_exp)
        return v
        
    def choose_action(self, state):

        with torch.no_grad():
            q = self.critic(state)
            v = self.getV(q).squeeze()
            dist = torch.exp((q-v)/ALPHA)
            dist = dist / torch.sum(dist)

            c = Categorical(dist)
            a = c.sample()
        return a.item()

    def update_parameter(self, batch):
        self.learn_steps = self.learn_steps +1
        if self.learn_steps % UPDATE_STEPS == 0:
            hard_update(self.critic_target, self.critic)

        # Concate data to tensors (No need for Variable)
        # Ensure tensors are moved to the correct device
        state_batch = to_device(torch.cat([b.state for b in batch]))
        action_batch = to_device(torch.cat([b.action for b in batch]).unsqueeze(1))
        reward_batch = to_device(torch.cat([b.reward for b in batch]).unsqueeze(1))
        done_batch = to_device(torch.cat([b.mask for b in batch]).unsqueeze(1))
        next_state_batch = to_device(torch.cat([b.next_state for b in batch]))

        # Get Q-values for current states and next states
        predict_q_values = self.critic(state_batch)

        # Disable gradient calculations for next state Q-values (inference mode)
        with torch.no_grad():
            next_q_values = self.critic_target(next_state_batch)

            # Get the Soft V for the next state (DQN target)
            next_v_value = self.getV(next_q_values)
        
            # Compute the TD target Q-value through Soft V
            # this no need to back propa, can use detach instaed of with torch.no_grad():
            target_q_value = reward_batch + GAMMA * next_v_value * (1 - done_batch)

        # Get the Q-values for the actions taken
        predict_q_value = predict_q_values.gather(1, action_batch.long())
        
        # Compute the loss
        value_loss = F.mse_loss(target_q_value, predict_q_value)
        # Optimize the model
        self.optim.zero_grad()
        value_loss.backward()
        self.optim.step()
        
        return value_loss

GAMMA = 0.99
REPLAY_MEMORY = 50000
BATCH = 16
UPDATE_STEPS = 4
ALPHA = 4

if __name__ == "__main__":
    env = gym.make('CartPole-v1')

    env.action_space.seed(1)
    env.reset(seed=1)
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1) # config for CPU
    torch.cuda.manual_seed(1) # config for GPU
    
    replay_buffer = ReplayMemory(REPLAY_MEMORY)
    softDqnAgent = SOFTDQNAGENT(env.observation_space.shape[0], env.action_space.n)
    
    begin_learn = False
    
    losses = []
    all_rewards = []
    ewma_reward = 0

    for episode_idx in range(10000):
        initState, _ = env.reset()
        state = torch.Tensor([initState]).to(deviceGPU)

        episode_reward = 0
        episode_loss = 0.0
        step = 0

        terminate = False
        truncated = False

        while not terminate and not truncated:
            step = step + 1
            action = softDqnAgent.choose_action(state)
            next_state, reward, terminate, truncated, _ = env.step(action)

            episode_reward += reward
            replay_buffer.push(
                state, 
                torch.tensor([action], device=deviceGPU), 
                torch.tensor([terminate], dtype=torch.float32, device=deviceGPU),
                torch.tensor([next_state], device=deviceGPU),
                torch.tensor([reward], device=deviceGPU)
                )

            if replay_buffer.__len__() > 128:
                if begin_learn is False:
                    print('learn begin!')
                    begin_learn = True
                
                training_batch = replay_buffer.sample(BATCH)
                
                value_loss = softDqnAgent.update_parameter(training_batch)
                episode_loss = episode_loss + value_loss.item()
                
            if terminate:
                break
            
            state = torch.tensor([next_state], device=deviceGPU)
        losses.append(episode_loss/step)        
        
        ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
        all_rewards.append(ewma_reward)
        
        if episode_idx % 100 == 0:
            print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}\t loss: {}'.format(episode_idx, step, 
                                                                                       episode_reward, ewma_reward, episode_loss/step))
    
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
    plt.savefig('SOFTDQN_plot-whole.png')
    plt.show()