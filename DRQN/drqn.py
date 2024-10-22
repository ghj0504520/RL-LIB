import sys
from typing import Dict, List, Tuple

import gym
import collections
import random, math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
import matplotlib.pyplot as plt


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


# Epsilon greedy decay as episodes increase
epsilon_start = 1.0
epsilon_final = 0.0001
epsilon_decay = 500

epsilon_by_episode = lambda episode_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * episode_idx / epsilon_decay)


# LSTM NN model
class LSTMCritic(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(LSTMCritic, self).__init__()
        self.lstmHidden_size = hidden_size

        self.layers1 = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU()
        )
        self.layers2 = nn.Sequential(
            nn.Linear(self.lstmHidden_size, hidden_size),
            nn.ReLU()
        )
        self.layers3 = nn.Sequential(
            nn.Linear(hidden_size, num_outputs)
        )
        self.lstm=nn.LSTM(hidden_size,self.lstmHidden_size, batch_first=True)
    
    def init_lstm_state(self,batch_size, training=None):
        if training is True:
            return torch.zeros([1, batch_size, self.lstmHidden_size]), torch.zeros([1, batch_size, self.lstmHidden_size])
        else:
            return torch.zeros([1, 1, self.lstmHidden_size]), torch.zeros([1, 1, self.lstmHidden_size])

    def forward(self, inputs, h, c):
        x = self.layers1(inputs)

        x, (new_h, new_c) = self.lstm(x,(h,c))

        #x = self.layers2(x)
        Q_value = self.layers3(x)
        return Q_value, new_h, new_c


# Replay buffer utility for LSTM history of Episode
Transition = namedtuple(
    'Transition', ('state', 'action', 'terminate', 'truncated', 'next_state', 'reward'))

class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, capacity=100, epi_capacity=500,
                        random_update=False, 
                        lookup_step=None):
        self.capacity = capacity
        self.epi_capacity = epi_capacity
        self.random_update = random_update # if False, sequential update
        self.lookup_step = lookup_step

        self.memory = []
        self.position = 0

    def put(self, episodeMem):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = episodeMem
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update: # Random upodate
            sampled_episodes = random.sample(self.memory, batch_size)
            
            min_step = self.epi_capacity

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode)) # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step: # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode)-self.lookup_step+1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode)-min_step+1) # sample buffer with minstep size
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################           
        else: # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]) # buffers, sequence_length

    def __len__(self):
        return len(self.memory)

class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.episodeMem=[]

    def put(self, *args):
        self.episodeMem.append(Transition(*args))

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        episodeBatch = self.episodeMem
        if random_update is True:
            episodeBatch = episodeBatch[idx:idx+lookup_step]
        return episodeBatch

    def __len__(self) -> int:
        return len(self.episodeMem)


# DRQN AGENT
class DRQNAGENT(object):
    def __init__(self, num_inputs, num_actions, hidden_size, lr, softUpdateFreq):
        super(DRQNAGENT, self).__init__()

        # num_inputs: env.observation_space.shape[0]
        # num_actions: env.action_space.n
        self.lstmCritic = LSTMCritic(hidden_size, num_inputs, num_actions).to(deviceGPU)
        self.lstmCritic_target = LSTMCritic(hidden_size, num_inputs, num_actions).to(deviceGPU)
        
        self.optim = optim.Adam(self.lstmCritic.parameters(), lr=lr)

        self.num_actions = num_actions
        self.softUpdateFreq = softUpdateFreq

        hard_update(self.lstmCritic_target, self.lstmCritic)
        
    def action_selection(self, state, h, c, epsilon):   
        # Without gradient tracking
        with torch.no_grad():
            q_value, h_, c_ = self.lstmCritic(state, h, c)
        
        if random.random() >= epsilon:
            # the result tuple of two output tensors (max, max_indices)
            # here, index is action
            action = q_value.max(2)[1].squeeze(0).item() # figure out dimension [batch, sequence, action]
        else:
            action = random.randrange(self.num_actions)
        return action , h_, c_

    def update_parameter(self, episode_idx, batch, seq_len):
        
        batch_size = len(batch)

        # Concate data to tensors (No need for Variable)
        # Ensure tensors are moved to the correct device
        state_batch = []
        action_batch  = []
        reward_batch = []
        done_batch = []
        truncated_batch = []
        next_state_batch  = []

        # concat data within each batch
        for epi_batch in batch:
            state_batch.append( to_device(torch.cat([b.state for b in epi_batch ])) )
            action_batch.append( to_device(torch.cat([b.action for b in epi_batch ])) )
            reward_batch.append( to_device(torch.cat([b.reward for b in epi_batch ])) )
            done_batch.append( to_device(torch.cat([b.terminate for b in epi_batch ])) )
            truncated_batch.append( to_device(torch.cat([b.truncated for b in epi_batch ])) )
            next_state_batch.append( to_device(torch.cat([b.next_state for b in epi_batch ])) )
        # compress list into one tensor
        state_batch = torch.stack(state_batch, dim=0)
        action_batch = torch.stack(action_batch, dim=0)
        reward_batch = torch.stack(reward_batch, dim=0)
        done_batch = torch.stack(done_batch, dim=0)
        truncated_batch = torch.stack(truncated_batch, dim=0)       
        next_state_batch = torch.stack(next_state_batch, dim=0)

        # Get predict Q-values for current states
        h_, c_ = self.lstmCritic.init_lstm_state(batch_size=batch_size, training=True)
        h_ = to_device(h_)
        c_ = to_device(c_)
        predict_q_values, _, _ = self.lstmCritic(state_batch, h_, c_)

        # Disable gradient calculations for next state target Q-values (inference mode)
        with torch.no_grad():
            h_target, c_target = self.lstmCritic_target.init_lstm_state(batch_size=batch_size, training=True)
            h_target = to_device(h_target)
            c_target = to_device(c_target)
            next_q_values, _, _  = self.lstmCritic_target(next_state_batch,h_target,c_target)
           
            # Get the maximum Q-value for the next state (DQN target)
            next_q_value = next_q_values.max(2)[0].view(batch_size, seq_len, -1) # figure out dimension [batch, sequence, action]
    
            target_q_value = reward_batch + gamma * next_q_value.squeeze(-1) * (1 - done_batch)

        # Get the predict Q-values for the actions taken
        predict_q_value = predict_q_values.gather(2, action_batch.unsqueeze(-1)) # figure out dimension [batch, sequence, action]
        
        # Compute the loss
        value_loss = F.mse_loss(target_q_value, predict_q_value.squeeze(-1))
        
        # Optimize the model
        self.optim.zero_grad()
        value_loss.backward()
        self.optim.step()
        
        if episode_idx % self.softUpdateFreq == 0:
            soft_update(self.lstmCritic_target, self.lstmCritic, tau)

        return value_loss



# Main training
env_id = "CartPole-v1"
env = gym.make(env_id)
total_episodes = 20000
batch_size = 64
learning_rate = 1e-3
gamma      = 0.99
tau = 1e-2
min_train_epi_num = 100 # Start moment to train the Q network
print_per_iter = 200
target_update_period = 4

# DRQN param
random_update = True# If you want to do random update instead of sequential update
lookup_step = 20 # If you want to do random update instead of sequential update
max_epi_len = 1000

losses = []
all_rewards = []
all_ewma_rewards = []
ewma_reward = 0

if __name__ == "__main__":
    
    drqnAgent = DRQNAGENT(env.observation_space.shape[0]-2, env.action_space.n,64,learning_rate,target_update_period)
    
    replay_buffer = EpisodeMemory(capacity=max_epi_len, epi_capacity=600, 
                                   random_update=random_update, 
                                   lookup_step=lookup_step)

    # Train
    for episode_idx in range(1, total_episodes + 1):
        initState, info = env.reset()
        partialState = initState[::2] # create partial observation, use only Position of Cart and Pole
        partialState = torch.Tensor([partialState]).to(deviceGPU)
        terminate = False
        truncated = False
        episode_reward=0.0
        episode_v_losses=0.0
        step = 0
        
        epsilon = epsilon_by_episode(episode_idx)
        episode_record = EpisodeBuffer() # store whole episode history sequence
        
        h, c = drqnAgent.lstmCritic.init_lstm_state(batch_size=batch_size, training=False)
        
        while not terminate and not truncated:
            step = step+1
            
            # Get action
            action, h, c = drqnAgent.action_selection(
                                    partialState.unsqueeze(0), 
                                    h.to(deviceGPU), c.to(deviceGPU),
                                    epsilon)

            # Do action
            next_state, reward, terminate, truncated, info = env.step(action)
            next_partialState = next_state[::2] # create partial observation,

            episode_record.put(
                partialState, 
                torch.tensor([action], device=deviceGPU), 
                torch.tensor([terminate], dtype=torch.float32, device=deviceGPU), 
                torch.tensor([truncated], dtype=torch.float32, device=deviceGPU), 
                torch.tensor([next_partialState], device=deviceGPU), 
                torch.tensor([reward], device=deviceGPU)
            )
            partialState = torch.Tensor([next_partialState]).to(deviceGPU)
            
            episode_reward += reward

            if len(replay_buffer) >= min_train_epi_num:
                training_batch, seq_len = replay_buffer.sample(batch_size)

                loss = drqnAgent.update_parameter(episode_idx, training_batch, seq_len)

                episode_v_losses += loss.item()
                
            if terminate:
                break
        
        replay_buffer.put(episode_record)

        ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward

        losses.append(episode_v_losses/step)
        all_rewards.append(episode_reward)
        all_ewma_rewards.append(ewma_reward)

        if episode_idx>=200 and episode_idx % print_per_iter == 0:
            print('Episode {}\tlength: {}\treward: {}\t ewma reward: {:.5f}\t loss: {:.5f}\t #_buffer : {}\t eps : {:.2f}%'.format(
                                                            episode_idx, step, 
                                                            episode_reward, ewma_reward, loss.item(),
                                                            len(replay_buffer), epsilon*100))
    env.close()


    plt.figure(figsize=(18, 6))
    # Plot for reward
    plt.subplot(1, 3, 1)
    plt.plot(all_rewards, label="Episode Reward")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Every Episodes')
    plt.grid(True)

    # Plot for EWMA
    plt.subplot(1, 3, 2)
    plt.plot(all_ewma_rewards, label="EWMA Reward")
    plt.xlabel('Episode')
    plt.ylabel('EWMA')
    plt.title('EWMA Reward Every Episodes')
    plt.grid(True)

    # Plot for loss
    plt.subplot(1, 3, 3)
    plt.plot(losses, label="Episode Loss", color='r')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss Every Episodes')
    plt.grid(True)

    plt.tight_layout()

    # Show legend
    plt.legend()

    # Save the plot as an image
    plt.savefig('DRQN_Target_plot-whole.png')
    plt.show()
