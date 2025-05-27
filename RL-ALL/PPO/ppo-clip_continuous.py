import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
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
    'Transition', ('state', 'action', 'terminate', 'truncated', 'next_state', 'reward', 'log_pb'))

class ReplayMemory(object):

    def __init__(self):
        self.memory = []
        self.position = 0

    def push(self, *args):
        self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = self.position + 1

    def clear(self):
        del self.memory[:]
        self.position = 0

    def __len__(self):
        return len(self.memory)


# CUDA GPU device usage utility
deviceGPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data):
    """Move a tensor or a collection of tensors to the specified device."""
    if isinstance(data, (list, tuple)):
        return [to_device(d, deviceGPU) for d in data]
    return data.to(deviceGPU)


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
 
class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs1, num_inputs2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu_head = nn.Linear(hidden_size, num_inputs2)
        self.log_std = nn.Parameter(torch.zeros(num_inputs2))  # standard deviation (std) often doesn’t need to depend on the input state
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mu_head, gain=0.01)
 
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = 2.0*torch.tanh(self.mu_head(x))    # bound between -2 ~ 2
        log_std = self.log_std
        if mu.shape[0]>1:                       # check and handle dimension
            log_std = self.log_std.expand_as(mu)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)  # Get the Gaussian distribution
        return dist 

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value
 

# Normalization Related
class RunningMeanStd:           # normalize and scale
                                # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
 
    def update(self, x):        # online algorithm
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)
 
    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
 
        return x

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)
 
    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x
 
    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

# GAE calculation with done state handling
class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done=True, next_value=None): # handle done
        advantages = []
        advantage = 0
        if done ==True:
            next_value = 0
        else:
            next_value = next_value
        for r,v in zip(reversed(rewards),reversed(values)):
            td_error = r + next_value*self.gamma - v
            #print(td_error,r,next_value,v)
            advantage = td_error + advantage*self.gamma*self.lambda_
            next_value = v
            advantages.insert(0,advantage.detach())
        advantages = to_device(torch.tensor(advantages))
        return advantages


# PPO Agent
class PPOCLIPAGENT:
    def __init__(self, num_inputs, action_space, gamma=0.99, hidden_size=64, batch_size=32,
                gae_lambda = 0.95, lr_a=3e-4, lr_c=3e-4, clip_param = 0.2, ppo_epoch = 10, entropy_coef = 0.01):

        self.actor = Actor(hidden_size, num_inputs, action_space).to(deviceGPU)
        self.critic = Critic(hidden_size, num_inputs).to(deviceGPU)
 
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_c)
        self.lr_a= lr_a
        self.lr_c= lr_c
 
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.ppo_epoch = ppo_epoch
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
 
    def select_action(self, state):
        state = state.unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(state)
        action = dist.sample()
        action = torch.clamp(action, -2, 2)
        return action.cpu().numpy().flatten(), dist.log_prob(action).sum(axis=-1).item()
 
    def lr_decay(self, ep):
        lr_a_now = self.lr_a * (1 - ep / 3000)
        lr_c_now = self.lr_c * (1 - ep / 3000)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_c_now
 
    def update1(self, batch,ep):
        states = to_device(torch.cat([b.state for b in batch]))
        actions = to_device(torch.cat([b.action for b in batch]))
        rewards = to_device(torch.cat([b.reward for b in batch]))
        dones = to_device(torch.cat([b.terminate for b in batch]).unsqueeze(1))
        next_states = to_device(torch.cat([b.next_state for b in batch]))
        old_log_probs = to_device(torch.cat([b.log_pb for b in batch]).unsqueeze(1))
        
        with torch.no_grad():  # adv and v_target have no gradient
            values = self.critic(states)
            next_values = self.critic(next_states)
            td_target = rewards + self.gamma * (1.0 - dones) * next_values
            if dones[-1] == True:
                advantage = GAE(self.gamma, self.gae_lambda,None)(rewards, values,True,None)
            else:
                advantage = GAE(self.gamma, self.gae_lambda,None)(rewards, values,False,next_values[-1])
            advantage = advantage.unsqueeze(1)

        old_log_probs = old_log_probs.detach()

        # Optimize policy for K epochs:
        for _ in range(self.ppo_epoch):
 
 
            new_act_dist = self.actor(states)
            new_act_dist_entropy = new_act_dist.entropy()  # shape(mini_batch_size X 1)
            new_log_probs = new_act_dist.log_prob(actions)
 
            ratios = torch.exp(new_log_probs - old_log_probs)  # shape(mini_batch_size X 1)

            surr1 = ratios * advantage  # Only calculate the gradient of 'a_logprob_now' in ratios
            surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantage
            policy_loss =  torch.mean(-torch.min(surr1, surr2)) - self.entropy_coef *  torch.mean(new_act_dist_entropy)  # Trick 5: policy entropy
            # Update actor
            self.actor_optimizer.zero_grad()
            policy_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
 
            v_s = self.critic(states)
            value_loss =  torch.mean(F.mse_loss(td_target, v_s))
            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        return policy_loss.mean().item(), value_loss.mean().item()
        #self.lr_decay(ep)

env = gym.make('Pendulum-v1')
env_evaluate = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_episodes = 4000
max_steps = 200
batch_size = 64

seed =0
env.action_space.seed(seed)
env.reset(seed=seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

agent = PPOCLIPAGENT(state_dim, action_dim, batch_size=batch_size)
ewma_reward = 0
ewma_reward_history = []
p_losses = []
v_losses = []

 
state_norm = Normalization(shape=state_dim)  # Trick 2:state normalization
replay_buffer = ReplayMemory()
reward_scaling = RewardScaling(shape=1, gamma=0.99)
for episode in range(max_episodes):
    initState, info = env.reset()
    state = torch.Tensor([initState]).to(deviceGPU)

    episode_reward = 0
    ep_p_losses=0
    ep_v_losses=0
    memory = []
    terminate = False
    truncate = False
    step = 0
    while terminate == False and truncate == False:
        step +=1
        action, log_prob = agent.select_action(state)
        next_state, reward, terminate, truncate, _ = env.step(action)
        reward = reward_scaling(reward) 
        episode_reward += reward

        replay_buffer.push(
                        state, 
                        torch.tensor([action], device=deviceGPU), 
                        torch.tensor([terminate], dtype=torch.float32, device=deviceGPU), 
                        torch.tensor([truncate], dtype=torch.float32, device=deviceGPU), 
                        torch.tensor([next_state], device=deviceGPU), 
                        torch.tensor([reward], dtype=torch.float32, device=deviceGPU),
                        torch.tensor([log_prob], dtype=torch.float32, device=deviceGPU)
                    )
        #print(replay_buffer.__len__())
        if replay_buffer.__len__() == 200:
            policy_loss, value_loss = agent.update1(replay_buffer.memory,episode)
            replay_buffer.clear()
            ep_p_losses+=policy_loss
            ep_v_losses+=value_loss
            #break

        state = torch.tensor([next_state], device=deviceGPU)
    
    ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
    ewma_reward_history.append(ewma_reward)
    p_losses.append(policy_loss/step)
    v_losses.append(value_loss/step)
 
    if episode % 1 == 0:
        print(f'Episode {episode}, {step},Reward: {episode_reward},EWMA Reward: {ewma_reward}')
    if (episode+1) % 200 == 0:
        for i in range (3):
            s, _ = env_evaluate.reset()
            er = 0
            done1 = False
            ss = 0
            while not done1:
                ss +=1
                s = torch.FloatTensor(s).to(deviceGPU).unsqueeze(0)
                with torch.no_grad():
                    dist = agent.actor(s)  # We use the deterministic policy during the evaluating

                action = dist.mean
                action = torch.clamp(action, -2, 2)
                s_, r, done1,ter, _ = env_evaluate.step(action.cpu().numpy().flatten())
                if ter:
                    done1 = True
                er += r
                s = s_
            print(f'evaluate step {ss},episode_reward {er}')

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
plt.savefig('PPO_CLIP_Continuous_plot-whole.png')
plt.show()