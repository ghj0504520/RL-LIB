import gym
import torch
import random
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym.utils.env_checker import check_env
from gym.wrappers import TimeLimit 
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


class Actor(nn.Module):

    def __init__(self, hidden_size, num_inputs1, num_inputs2):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_inputs1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_inputs2),
            nn.Softmax(dim=-1) # change to 1
        )

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist


class Critic(nn.Module):

    def __init__(self, hidden_size, num_inputs1):
        super(Critic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.critic(state)


# General Advantage Estimation facility
class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done=True, next_value=None):
        advantages = []
        advantage = 0
        if done ==True:
            next_value = 0
        else:
            next_value = next_value
        for r,v in zip(reversed(rewards),reversed(values)):
            td_error = r + next_value*self.gamma - v
            advantage = td_error + advantage*self.gamma*self.lambda_
            next_value = v
            advantages.insert(0,advantage.detach())
        advantages = to_device(torch.tensor(advantages))
        return advantages


class PPOCLIPAGENT():
    def __init__(self, num_inputs, action_space, gamma=0.995, hidden_size=128, batch_size=32,
                gae_lambda = 0.999, lr_a=1e-4, lr_c=1e-3, clip_param = 0.2, ppo_epoch = 10):
        self.gamma = gamma
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        
        self.actor = Actor(hidden_size, num_inputs, action_space.n).to(deviceGPU)
        self.critic = Critic(hidden_size, num_inputs).to(deviceGPU) 
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

    def select_action(self, state):
        #state = torch.tensor(state, dtype=torch.float).to(deviceGPU)
        state = state.unsqueeze(0)
        action_dist = self.actor(state)
        
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action).item()

    def update(self, batch):
        states = to_device(torch.cat([b.state for b in batch]))
        actions = to_device(torch.cat([b.action for b in batch]).unsqueeze(1))
        rewards = to_device(torch.cat([b.reward for b in batch]).unsqueeze(1))
        dones = to_device(torch.cat([b.terminate for b in batch]).unsqueeze(1))
        next_states = to_device(torch.cat([b.next_state for b in batch]))
        old_log_probs = to_device(torch.cat([b.log_pb for b in batch]).unsqueeze(1))

        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1-dones)
        
            values = self.critic(states)
            if dones[-1] == True:
                advantage = GAE(self.gamma, self.gae_lambda,None)(rewards, values,True,None)
            else:
                fin_next_value = self.critic(next_states[-1])
                advantage = GAE(self.gamma, self.gae_lambda,None)(rewards, values,False,fin_next_value)
            
            advantage = advantage.unsqueeze(1)
        old_log_probs = old_log_probs.detach()
        
        for _ in range(self.ppo_epoch):
            new_act_dist = self.actor(states)
            new_log_probs = new_act_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
           
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage 
            policy_loss = torch.mean(-torch.min(surr1, surr2))
            value_loss = torch.mean(F.mse_loss(self.critic(states), td_target))


            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()
        return policy_loss.item(), value_loss.item()

if __name__ == "__main__":
    def moving_average(a, window_size):
        cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
        middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        r = np.arange(1, window_size-1, 2)
        begin = np.cumsum(a[:window_size-1])[::2] / r
        end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
        return np.concatenate((begin, middle, end))

    def set_seed(env, seed=42):
        env.action_space.seed(seed)
        env.reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    num_episodes = 500
    gamma = 0.98
    gae_lambda = 0.95
    hidden_size = 128
    replay_size = 1000
    batch_size = 32
    clip_param = 0.2
    ppo_epoch = 10
    updates_per_step = 1
    print_freq = 1
    ewma_reward = 0
    ewma_reward_history = []
    p_losses = []
    v_losses = []
    total_numsteps = 0
    updates = 0
    lr_a=1e-3
    lr_c=1e-2

    # build environment
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    set_seed(env, 0)

    replay_buffer = ReplayMemory()

    # build agent
    agent = PPOCLIPAGENT( env.observation_space.shape[0], env.action_space, gamma=gamma, hidden_size=hidden_size, batch_size=batch_size,
                gae_lambda = gae_lambda , lr_a=lr_a, lr_c=lr_c, clip_param = clip_param, ppo_epoch = ppo_epoch)

    return_list = []
    for i in range(30):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_reward = 0

                initState, info = env.reset()
                state = torch.Tensor([initState]).to(deviceGPU)
    
                terminate = False
                truncated = False
                
                while terminate == False and truncated == False:
                    action, log_prob = agent.select_action(state)
                    next_state, reward, terminate, truncated, info = env.step(action)

                    done = terminate
                    replay_buffer.push(
                        state, 
                        torch.tensor([action], device=deviceGPU), 
                        torch.tensor([done], dtype=torch.float32, device=deviceGPU), 
                        torch.tensor([truncated], dtype=torch.float32, device=deviceGPU), 
                        torch.tensor([next_state], device=deviceGPU), 
                        torch.tensor([reward], dtype=torch.float32, device=deviceGPU),
                        torch.tensor([log_prob], dtype=torch.float32, device=deviceGPU)
                    )
                    
                    state = torch.tensor([next_state], device=deviceGPU)
                    episode_reward += reward
                                        
                    if done:
                        break
                    #env.render()

                policy_loss, value_loss = agent.update(replay_buffer.memory)
                ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
                ewma_reward_history.append(ewma_reward)
                p_losses.append(policy_loss)
                v_losses.append(value_loss)

                return_list.append(episode_reward)
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % episode_reward,
                    'ave return':
                    '%.3f' % np.mean(return_list[-10:]),
                    'ewma return':
                    '%.3f' % ewma_reward
                })
                pbar.update(1)
                replay_buffer.clear()

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
    plt.savefig('PPO_CLIP_Discrete_plot-whole.png')
    plt.show()