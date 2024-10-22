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


# Replay buffer utility
Transition = namedtuple(
    'Transition', ('state', 'action', 'terminate', 'truncated', 'next_state', 'reward'))


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


# NN model
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

    def sample_action(self, obs, h,c, epsilon):
        output = self.forward(obs, h,c)

        if random.random() < epsilon:
            return random.randint(0,1), output[1], output[2]
        else:
            #print(output[0])
            #print(output[0].argmax())
            #print(output[0].max(1)[0].squeeze(0))
            #print(output[0].max(1)[1].squeeze(0))
            #print(output[0].max(2)[0].squeeze(0))
            #print(output[0].max(2)[1].squeeze(0).item())
            return output[0].max(2)[1].squeeze(0).item(), output[1] , output[2]
    
    def init_lstm_state(self,batch_size, training=None):
        if training is True:
            return torch.zeros([1, batch_size, self.lstmHidden_size]), torch.zeros([1, batch_size, self.lstmHidden_size])
        else:
            return torch.zeros([1, 1, self.lstmHidden_size]), torch.zeros([1, 1, self.lstmHidden_size])

    def forward(self, inputs, h, c):
        #print(inputs.shape)
        #print(h.shape)
        #print(c.shape)
        #if h.dim() == 3 and c.dim() == 3 and inputs.size(0) == 1:  # unbatched case
        #    h = h.squeeze(0)
        #    c = c.squeeze(0)
        
        x = self.layers1(inputs)

        x, (new_h, new_c) = self.lstm(x,(h,c))

        #x = self.layers2(x)
        Q_value = self.layers3(x)
        return Q_value, new_h, new_c

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


def train(q_net=None, target_q_net=None, episode_memory=None,
          device=None, 
          optimizer = None,
          batch_size=1,
          learning_rate=1e-3,
          gamma=0.99):

    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples, seq_len = episode_memory.sample(batch_size)

    state_batch = []
    action_batch  = []
    reward_batch = []
    done_batch = []
    truncated_batch = []
    next_state_batch  = []

    for epi_batch in samples:
            state_batch.append( to_device(torch.cat([b.state for b in epi_batch ])) )
            action_batch.append( to_device(torch.cat([b.action for b in epi_batch ])) )
            reward_batch.append( to_device(torch.cat([b.reward for b in epi_batch ])) )
            done_batch.append( to_device(torch.cat([b.terminate for b in epi_batch ])) )
            truncated_batch.append( to_device(torch.cat([b.truncated for b in epi_batch ])) )
            next_state_batch.append( to_device(torch.cat([b.next_state for b in epi_batch ])) )
    state_batch = torch.stack(state_batch, dim=0)
    action_batch = torch.stack(action_batch, dim=0)
    reward_batch = torch.stack(reward_batch, dim=0)
    done_batch = torch.stack(done_batch, dim=0)
    truncated_batch = torch.stack(truncated_batch, dim=0)       
    next_state_batch = torch.stack(next_state_batch, dim=0)
    
    h_target, c_target = target_q_net.init_lstm_state(batch_size=batch_size, training=True)

    q_target, _, _ = target_q_net(next_state_batch, h_target.to(device), c_target.to(device))

    q_target_max = q_target.max(2)[0].view(batch_size,seq_len,-1).detach()
    #print(next_q_value)
    #print(next_q_value.squeeze(-1).shape)
    # Compute the TD target Q-value
    # this no need to back propa, can use detach instaed of with torch.no_grad():
    #print(reward_batch.shape)
    #print(q_target_max.shape)
    #print(done_batch.shape)
    targets = reward_batch + gamma*q_target_max.squeeze(-1)*(1-done_batch)
#    print()
#    print("trtrtr")
    h, c = q_net.init_lstm_state(batch_size=batch_size, training=True)
    q_out, _, _ = q_net(state_batch, h.to(device), c.to(device))
#    print()
    #print(q_out.shape)
    #print(action_batch.shape)
#    print()
    # predict_q_values.gather(2, action_batch.unsqueeze(-1))
    #print(q_out.shape)
    #print(action_batch.unsqueeze(-1).shape)
    #print(q_out.gather(2, action_batch.unsqueeze(-1)).shape)
    q_a = q_out.gather(2, action_batch.unsqueeze(-1))
    
    # Multiply Importance Sampling weights to loss   
    #print(q_a.shape)
    #print(targets.squeeze(-1).shape)     
    loss = F.smooth_l1_loss(q_a.squeeze(-1), targets)
    
    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


if __name__ == "__main__":

    # Env parameters
    model_name = "DRQN_POMDP_Random"
    env_name = "CartPole-v1"

    # Set gym environment
    env = gym.make(env_name)
    

    # default `log_dir` is "runs" - we'll be more specific here
    # Set parameters
    batch_size = 64
    learning_rate = 1e-3
    buffer_len = int(100000)
    min_epi_num = 100 # Start moment to train the Q network
    episodes = 20000
    print_per_iter = 200
    target_update_period = 4
    eps_start = 0.1
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2
    max_step = 2000

    # DRQN param
    random_update = True# If you want to do random update instead of sequential update
    lookup_step = 20 # If you want to do random update instead of sequential update
    max_epi_len = 1000
    max_epi_step = max_step

    

    # Create Q functions
    Q = LSTMCritic(64,env.observation_space.shape[0]-2, 
              env.action_space.n).to(deviceGPU)
    Q_target = LSTMCritic(64,env.observation_space.shape[0]-2, 
                     env.action_space.n).to(deviceGPU)

    hard_update(Q_target,Q)

    # Set optimizer
    score_sum = 0
    all_rewards = []
    losses = []
    ewma_reward = 0
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    
    episode_memory = EpisodeMemory(capacity=1000, epi_capacity=600, 
                                   random_update=random_update, 
                                   lookup_step=lookup_step)

    # Train
    for episode_idx in range(episodes):
        s, _ = env.reset()
        obs = s[::2] # Use only Position of Cart and Pole
        terminate = False
        truncated = False
        
        epsilon = epsilon_by_episode(episode_idx)
        episode_record = EpisodeBuffer()
        h, c = Q.init_lstm_state(batch_size=batch_size, training=False)
        score_sum=0.0
        step = 0
        

        while not terminate and not truncated:
            step = step+1

            # Get action
            a, h, c = Q.sample_action(torch.from_numpy(obs).float().to(deviceGPU).unsqueeze(0).unsqueeze(0), 
                                              h.to(deviceGPU), c.to(deviceGPU),
                                              epsilon)

            # Do action
            s_prime, r, terminate,truncated , info = env.step(a)
            obs_prime = s_prime[::2]

            # make data
            done_mask = 0.0 if terminate else 1.0

            episode_record.put(
                torch.Tensor([obs]).to(deviceGPU), 
                torch.tensor([a], device=deviceGPU), 
                torch.tensor([terminate], dtype=torch.float32, device=deviceGPU), 
                torch.tensor([truncated], dtype=torch.float32, device=deviceGPU), 
                torch.tensor([obs_prime], device=deviceGPU), 
                torch.tensor([r], device=deviceGPU)
            )
            obs = obs_prime
            
            score_sum += r

            if len(episode_memory) >= min_epi_num:
                loss = train(Q, Q_target, episode_memory, deviceGPU, 
                        optimizer=optimizer,
                        batch_size=batch_size,
                        learning_rate=learning_rate)
                losses.append(loss.item())
                if (step+1) % target_update_period == 0:
                    soft_update(Q_target, Q,tau)
                
            if terminate:
                break
        
        episode_memory.put(episode_record)
        
        ewma_reward = 0.05 * score_sum + (1 - 0.05) * ewma_reward
        all_rewards.append(score_sum)
        if episode_idx>=200 and episode_idx % 200 == 0:
            print('Episode {}\tlength: {}\treward: {}\t ewma reward: {:.5f}\t loss: {:.5f} n_buffer : {} eps : {:.2f}%'.format(
                                                            episode_idx, step, 
                                                            score_sum, ewma_reward, loss.item(),
                                                            len(episode_memory), epsilon*100))

        # Log the reward
        
    env.close()

    chunk_size = 1  # Every 1 episodes
    num_chunks = len(all_rewards) // chunk_size  # Number of chunks


    # Plotting utility
    # Calculate average reward for every 200 episodes
    average_rewards = [sum(all_rewards[i * chunk_size:(i + 1) * chunk_size]) / chunk_size for i in range(num_chunks)]
    #average_loss = [sum(losses[i * chunk_size:(i + 1) * chunk_size]) / chunk_size for i in range(num_chunks)]

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
    #plt.subplot(1, 2, 2)
    #plt.plot(range(1, num_chunks + 1), average_loss, label="Average Loss", color='r')
    #plt.xlabel('Episode')
    #plt.ylabel('Loss')
    #plt.title('Average Loss Every Episodes')
    #plt.grid(True)

    plt.tight_layout()

    # Show legend
    plt.legend()

    # Save the plot as an image
    plt.savefig('DRQN_Target_plot-whole.png')
    plt.show()
