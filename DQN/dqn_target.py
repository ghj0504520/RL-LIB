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

# Epsilon greedy decay as episodes increase
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_episode = lambda episode_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * episode_idx / epsilon_decay)

# NN model
class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Critic, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )

    def forward(self, inputs):
        return self.layers(inputs)


# DQN with Target Separation AGENT
class DQNTARGETAGENT(object):
    def __init__(self, num_inputs, num_actions, lr_c=1e-3):
        super(DQNTARGETAGENT, self).__init__()

        # num_inputs: env.observation_space.shape[0]
        # num_actions: env.action_space.n
        self.critic = Critic(256, num_inputs, num_actions)
        self.critic_target = Critic(256, num_inputs, num_actions)
        
        self.optim = optim.Adam(self.critic.parameters(), lr=lr_c)
        self.num_actions = num_actions

        hard_update(self.critic_target, self.critic)

    def action_selection(self, state, epsilon):
        if random.random() > epsilon:

            # if your model has Dropout or BatchNorm, needing to set this
            self.critic.eval()
            # figure out, state is already in device
            # state = to_device(state)
            if len(state.shape) == 1:  # If it's a single state without batch dimension
                state = state.unsqueeze(0)
            
            # Without gradient tracking
            with torch.no_grad():
                 q_value = self.critic(state)
            
            # the result tuple of two output tensors (max, max_indices)
            # here, index is action
            action  = q_value.max(1)[1].squeeze(0).item()
        else:
            action = random.randrange(self.num_actions)
        return action

    def update_parameter(self, episode_idx, batch):

        # Concate data to tensors (No need for Variable)
        # Ensure tensors are moved to the correct device
        state_batch = to_device(torch.cat([b.state for b in batch]))
        action_batch = to_device(torch.cat([b.action for b in batch]))
        reward_batch = to_device(torch.cat([b.reward for b in batch]))
        done_batch = to_device(torch.cat([b.mask for b in batch]))
        next_state_batch = to_device(torch.cat([b.next_state for b in batch]))


        # Get Q-values for current states and next states
        predict_q_values = self.critic(state_batch)

        # Disable gradient calculations for next state Q-values (inference mode)
        with torch.no_grad():
            next_q_values = self.critic_target(next_state_batch)

            # Get the maximum Q-value for the next state (DQN target)
            next_q_value = next_q_values.max(1)[0]
        
            # Compute the TD target Q-value
            # this no need to back propa, can use detach instaed of with torch.no_grad():
            target_q_value = reward_batch + gamma * next_q_value * (1 - done_batch)

        # Get the Q-values for the actions taken
        predict_q_value = predict_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Compute the loss
        value_loss = F.mse_loss(target_q_value, predict_q_value)
        # Optimize the model
        self.optim.zero_grad()
        value_loss.backward()
        self.optim.step()
        
        if episode_idx % 100 == 0:
            hard_update(self.critic_target, self.critic)

        return value_loss


# Main training
env_id = "CartPole-v1"
env = gym.make(env_id)
total_episodes = 20000
batch_size = 128
gamma      = 0.99
tau = 0.0002
lr_c = 3e-3

losses = []
all_rewards = []
ewma_reward = 0

dqnTargetAgent = DQNTARGETAGENT(env.observation_space.shape[0], env.action_space.n, lr_c)

replay_buffer = ReplayMemory(100000)

for episode_idx in range(1, total_episodes + 1):
    initState, info = env.reset()
    state = torch.Tensor([initState]).to(deviceGPU)
    epsilon = epsilon_by_episode(episode_idx)
    done = False
    episode_reward = 0
    step = 0
    while not done:
        step = step+1
        action = dqnTargetAgent.action_selection(state, epsilon) #select action from updated q net
        next_state, reward, terminate, truncated, info = env.step(action)
        
        done = terminate or truncated
        replay_buffer.push(
            state, 
            torch.tensor([action], device=deviceGPU), 
            torch.tensor([done], dtype=torch.float32, device=deviceGPU), 
            torch.tensor([next_state], device=deviceGPU), 
            torch.tensor([reward], device=deviceGPU)
        )
    
        state = torch.tensor([next_state], device=deviceGPU)  # Ensure next state is on the correct device
        episode_reward += reward

        if done:
            break

    ############################################# Per-episode training instead of per-step
    if(replay_buffer.__len__() >= batch_size):
        training_batch = replay_buffer.sample(batch_size)

        loss = dqnTargetAgent.update_parameter(episode_idx, training_batch)
        losses.append(loss.item())

    ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
        
    all_rewards.append(ewma_reward)

    if episode_idx % 200 == 0:
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}\t loss: {}'.format(episode_idx, step, 
                                                                                       episode_reward, ewma_reward, loss.item()))
    
    # Early termination
    #if ewma_reward > env.spec.reward_threshold:
    #        print("Solved! Running reward is now {} and "
    #        "the last episode runs to {} time steps!".format(ewma_reward, step))
    #        break

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
plt.savefig('DQN_Target_plot-whole.png')
plt.show()
