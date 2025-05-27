import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
import matplotlib.pyplot as plt

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value','entropy'])


# CUDA GPU device usage utility
deviceGPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data):
    #print("GPU: ",torch.cuda.is_available())
    """Move a tensor or a collection of tensors to the specified device."""
    if isinstance(data, (list, tuple)):
        return [to_device(d, deviceGPU) for d in data]
    return data.to(deviceGPU)


## NN model for Actor and Critic
class ActorCritic(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(ActorCritic, self).__init__()

        # Common layers
        self.commonLayer = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU()            
        )
        # Value stream
        self.valueLayer = nn.Sequential(
            nn.Linear(hidden_size, 1)  # Only one value for the state V(s)
        )
        # Actor stream
        self.actorLayer = nn.Sequential(
            nn.Linear(hidden_size, num_outputs),  # Action prob of state V(s)
            nn.Softmax(dim=1) # here, directly apply softmax to obatin log probability
        )

    def forward(self, inputs):
        features = self.commonLayer(inputs)

        # Compute the value and action prob
        action_prob = self.actorLayer(features)
        state_value = self.valueLayer(features)
        action_dist  = Categorical(action_prob)

        return action_dist, state_value


class A2CAGENT(object):
    def __init__(self, num_inputs, action_dim, gamma=0.999, hidden_size=128, lr_ac=0.01):
        super(A2CAGENT, self).__init__()
        
        self.num_inputs = num_inputs
        self.action_dim = action_dim        
        self.gamma = gamma

        self.actorCritic = ActorCritic(hidden_size, self.num_inputs, self.action_dim).to(deviceGPU)
        self.actorCritic_optim = optim.Adam(self.actorCritic.parameters(), lr=lr_ac)
        
        # action & reward memory per episode
        self.saved_actions = []
        self.rewards = []

    def select_action(self, state):
        # cannot disable gradient
        action_dis, state_value = self.actorCritic(state)
        action = action_dis.sample()

        # save to action buffer
        self.saved_actions.append(SavedAction(action_dis.log_prob(action), 
                                              state_value, action_dis.entropy()))
        return action.item()

    def calculate_loss(self, entropy_loss_weight = 0.01):
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []
        
        with torch.no_grad():
            for t in reversed(range(len(self.rewards))):
                R = self.rewards[t] + R * self.gamma
                returns.insert(0, R)
            # Convert returns to a tensor and move to the appropriate device
            returns = to_device(torch.tensor(returns).unsqueeze(1))

        log_probs = [action.log_prob for action in saved_actions] # get log probs list from saved_actions
        values = [action.value for action  in saved_actions] # get values list from saved_actions
        entropies = [action.entropy for action  in saved_actions]

        log_probs = to_device(torch.stack(log_probs, dim=0)) # concat to linear
        values = to_device(torch.stack(values, dim=0).squeeze(-1)) # concat to linear; squeeze(-1), incorrect results due to broadcast
        entropies = to_device(torch.stack(entropies, dim=0))

        advantage = (returns-values).detach()
        policy_losses = (-log_probs * advantage) # maximize reward = minimize negative reward
        value_losses = F.mse_loss(values, returns) # mse loss
        entropy_losses = - entropies

        vallue_loss_weight = 1 # add weight to loss, but here just use 1
        #print(policy_losses)
        #print(value_losses)

        # weighted loss
        loss = policy_losses.mean() + vallue_loss_weight * value_losses.mean() + entropy_loss_weight * entropy_losses.mean()
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


def train(lr=0.01):
    hidden_size = 128
    gamma = 0.999
    num_episodes = 10000
    lr_ac = lr

    # Extract the dimensionality of state and action spaces
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]
    
    # Instantiate the agebt and the optimizer
    a2cAgent = A2CAGENT(observation_dim, action_dim, gamma, hidden_size, lr_ac)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(a2cAgent.actorCritic_optim, step_size=100, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    ewma_reward_history = []
    losses = []
    
    # run inifinitely many episodes
    for episode_idx in range(num_episodes):
        # reset environment and episode reward
        initState, info = env.reset()
        state = torch.Tensor([initState]).to(deviceGPU)
        ep_reward = 0
        t = 0
        scheduler.step()
        
        # For each episode
        for t in range(10000):
            action = a2cAgent.select_action(state)
            next_state, reward, terminate, truncated ,_ = env.step(action)

            a2cAgent.rewards.append(reward)
            ep_reward += reward
            if terminate or truncated:
                break
            state = torch.tensor([next_state], device=deviceGPU)

        a2cAgent.actorCritic_optim.zero_grad()   # each time need reset
        loss = a2cAgent.calculate_loss()
        loss.backward()
        a2cAgent.actorCritic_optim.step()
        a2cAgent.clear_memory()    # clean memory
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward

        losses.append(loss.detach().cpu().numpy())
        ewma_reward_history.append(ewma_reward)

        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(episode_idx, t, ep_reward, ewma_reward))
        
        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        #if ewma_reward > env.spec.reward_threshold:
        #    print("Solved! Running reward is now {} and "
        #          "the last episode runs to {} time steps!".format(ewma_reward, t))
        #    break
    
    # Plot the losses and EWMA reward after training
    plt.figure(figsize=(12, 6))
    
    # Plot policy loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.title('Loss Over Episodes')
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot EWMA reward
    plt.subplot(1, 2, 2)
    plt.plot(ewma_reward_history, label='EWMA Reward', color='green')
    plt.title('EWMA Reward Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('EWMA Reward')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save the plot as an image
    plt.savefig('A2C_plot-whole.png')
    #plt.show()

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    #random_seed = 10  
    lr = 0.01
    env = gym.make('CartPole-v1')
    #env.seed(random_seed)  
    #torch.manual_seed(random_seed)  
    train(lr)
    #test(f'CartPole_{lr}.pth')
