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

# Prioritized Replay buffer utility
Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class Critic(nn.Module):
    def __init__(self, hidden_dim, n_states,n_actions):
        """ 初始化q网络，为全连接网络
        """
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.layers(x)#self.fc3(x)

class SumTree(object):
    

    def __init__(self, capacity):
        self.data_pointer = 0
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, transition):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = transition  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        #self.data_pointer += 1
        #if self.data_pointer >= self.capacity:  # replace when exceed the capacity
        #    self.data_pointer=0
        self.data_pointer = (self.data_pointer+1)%self.capacity

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return int(self.tree[0]) # the root

class PrioritizedReplayMemory:
    '''ReplayTree for the per(Prioritized Experience Replay) DQN. 
    '''
    def __init__(self, capacity):
        self.capacity = capacity # the capacity for memory replay
        self.tree = SumTree(capacity)
        self.cnt=0

        ## hyper parameter for calculating the importance sampling weight
        self.beta_increment_per_sampling = 0.001
        self.alpha = 0.6
        self.beta = 0.4
        self.epsilon = 0.01 
        self.abs_err_upper = 1.

    def __len__(self):
        ''' return the num of storage
        '''
        #print((self.tree.n_entries), self.cnt)
        return self.cnt

    def push(self, error, sample):
        '''Push the sample into the replay according to the importance sampling weight
        '''
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(p, sample)         
        if self.cnt < self.capacity:
            self.cnt=self.cnt+1

    def sample(self, batch_size):
        '''This is for sampling a batch data and the original code is from:
        https://github.com/rlcode/per/blob/master/prioritized_memory.py
        '''
        segment = self.tree.total() / batch_size

        priorities = []
        batch = []
        idxs = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i+1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get_leaf(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
            prob = p / self.tree.total()

        sampling_probabilities = np.array(priorities) / self.tree.total()
        idx_weight = np.power(self.cnt * sampling_probabilities, -self.beta)
        idx_weight /= idx_weight.max()

        return batch, idxs, idx_weight
    
    def batch_update(self, tree_idx, abs_errors):
        '''Update the importance sampling weight
        '''
        abs_errors += self.epsilon
        #print(abs_errors)
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        new_priprity = clipped_errors ** self.alpha
        for ti, p in zip(tree_idx, new_priprity):
            self.tree.update(ti, p)
        
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

# DQN with Target Separation AGENT
class DQNTARGETAGENT(object):
    def __init__(self, num_inputs, num_actions):
        super(DQNTARGETAGENT, self).__init__()

        # num_inputs: env.observation_space.shape[0]
        # num_actions: env.action_space.n
        self.critic = Critic(256, num_inputs, num_actions).to(deviceGPU)
        self.critic_target = Critic(256, num_inputs, num_actions).to(deviceGPU)
        
        self.optim = optim.Adam(self.critic.parameters(), lr = 0.0001 )
        self.num_actions = num_actions
        self.update_cnt =0
        self.replay_buffer = PrioritizedReplayMemory(10000)
        hard_update(self.critic_target, self.critic)

    def store_transition(self,s,a,done,s_,r):
        policy_val =self.critic(s).gather(1, a.unsqueeze(1)).squeeze(1)
        target_val =self.critic_target(s_)
        transition = Transition(s,a,done,s_,r)
        #print(policy_val,r,torch.max(target_val),(1-done))
        td_error = abs(policy_val - r -gamma*torch.max(target_val)*(1-done))
        self.replay_buffer.push(td_error.detach().cpu(), transition)  # 添加经验和初始优先级
        return self.replay_buffer.__len__()

    def action_selection(self, state, epsilon, infer=False):
        if infer == True:
            # if your model has Dropout or BatchNorm, needing to set this
            #print("Infer")
            self.critic.eval()
            with torch.no_grad():
                 q_value = self.critic(state)
            # the result tuple of two output tensors (max, max_indices)
            # here, index is action
            action  = q_value.max(1)[1].squeeze(0).item()
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

    def update_parameter(self, episode_idx, batch_size):
        self.update_cnt = self.update_cnt+1
        batch, idxs, idx_weight = self.replay_buffer.sample(batch_size)
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
        value_loss = (to_device(torch.FloatTensor(idx_weight)) * F.mse_loss(target_q_value, predict_q_value)).mean()
        # Optimize the model
        self.optim.zero_grad()
        value_loss.backward()
        for param in self.critic.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

        abs_errors = torch.abs(predict_q_value - target_q_value).detach().cpu().numpy().squeeze()
        self.replay_buffer.batch_update(idxs, abs_errors)
        #print(idx,self.update_cnt)
        if episode_idx % 100 == 0:
            #print("hard",self.update_cnt)
            hard_update(self.critic_target, self.critic)

        return value_loss


# Main training
env_id = "CartPole-v1"
env = gym.make(env_id)
total_episodes = 6000
batch_size = 256
gamma      = 0.95
tau = 0.0002

env.action_space.seed(1)
env.reset(seed=1)
np.random.seed(1)
random.seed(1)
torch.manual_seed(1) # config for CPU
torch.cuda.manual_seed(1) # config for GPU

losses = []
all_rewards = []
ewma_reward = 0

dqnTargetAgent = DQNTARGETAGENT(env.observation_space.shape[0], env.action_space.n)
buf_len=0
for episode_idx in range(1, total_episodes + 1):
    initState, info = env.reset()
    state = torch.Tensor([initState]).to(deviceGPU)
    epsilon = epsilon_by_episode(episode_idx)

    terminate = False
    truncated = False
    episode_reward = 0
    step = 0

    while not terminate and not truncated:
        step = step+1
        action = dqnTargetAgent.action_selection(state, epsilon) #select action from updated q net
        next_state, reward, terminate, truncated, info = env.step(action)
        
        done = terminate
        buf_len = dqnTargetAgent.store_transition(
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
    if(buf_len >= batch_size):
        loss = dqnTargetAgent.update_parameter(episode_idx, batch_size)
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

print("testing")
rewards = []  
steps = []
for episode_idx in range(20):
    ep_reward = 0  
    ep_step = 0
    initState,_ = env.reset() 
    state = torch.Tensor([initState]).to(deviceGPU)
    terminate = False
    truncated = False

    while not terminate and not truncated:
        ep_step+=1
        action = dqnTargetAgent.action_selection(state,epsilon=0,infer=True)  
        next_state, reward, terminate, truncated, _ = env.step(action)  
        state = torch.tensor([next_state], device=deviceGPU)
        ep_reward += reward 
        if terminate or truncated:
            break
    rewards.append(ep_reward)
    print(f"{episode_idx+1}/{20}，reward:{ep_reward:.2f}")
print("fin")
env.close()


chunk_size = 1  # Every 1 episodes
num_chunks = len(all_rewards) // chunk_size  # Number of chunks


# Plotting utility
# Calculate average reward for every 200 episodes
average_rewards = [sum(all_rewards[i * chunk_size:(i + 1) * chunk_size]) / chunk_size for i in range(num_chunks)]
average_loss = [sum(losses[i * chunk_size:(i + 1) * chunk_size]) / chunk_size for i in range(num_chunks)]

# Plot the average reward and loss every 200 episodes
plt.figure(figsize=(18, 6))

# Plot for reward
plt.subplot(1, 3, 1)
plt.plot(range(1, num_chunks + 1), average_rewards, label="Average Reward")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Average Reward Every Episodes')
plt.grid(True)

# Plot for loss
plt.subplot(1, 3, 2)
plt.plot(range(1, num_chunks + 1), average_loss, label="Average Loss", color='r')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Average Loss Every Episodes')
plt.grid(True)

# Plot for infer
plt.subplot(1, 3, 3)
plt.plot(rewards, label="Evaluating Reward", color='r')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Every Evaluating Episodes')
plt.grid(True)

plt.tight_layout()

# Show legend
plt.legend()

# Save the plot as an image
plt.savefig('DQN_Target_plot-whole.png')
plt.show()
