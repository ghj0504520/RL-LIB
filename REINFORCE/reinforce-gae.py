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
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


# CUDA GPU device usage utility
deviceGPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data):
    """Move a tensor or a collection of tensors to the specified device."""
    if isinstance(data, (list, tuple)):
        return [to_device(d, deviceGPU) for d in data]
    return data.to(deviceGPU)


## NN model for Actor and Critic
class ActorCritic(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(ActorCritic, self).__init__()

        self.double()

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
            nn.Linear(hidden_size, num_outputs)  # Action prob of state V(s)
        )

    def forward(self, inputs):
        features = self.commonLayer(inputs)

        # Compute the value and action prob
        action_prob = self.actorLayer(features)
        state_value = self.valueLayer(features)
        
        return action_prob, state_value


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


class REINFORCEAGENT(object):
    def __init__(self, num_inputs, action_dim, gamma=0.999, gae_lambda=0.999, hidden_size=128, lr_ac=0.01):
        super(REINFORCEAGENT, self).__init__()
        
        self.num_inputs = num_inputs
        self.action_dim = action_dim        
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.actorCritic = ActorCritic(hidden_size, self.num_inputs, self.action_dim).to(deviceGPU)
        self.actorCritic_optim = optim.Adam(self.actorCritic.parameters(), lr=lr_ac)
        
        # action & reward memory per episode
        self.saved_actions = []
        self.dones = []
        self.rewards = []
        self.next_states = []

    def select_action(self, state):
        # cannot disable gradient
        action_prob, state_value = self.actorCritic(state)
        m = Categorical(logits=action_prob) # log prob
        action = m.sample()

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    def calculate_loss(self):
        
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

        log_probs = to_device(torch.stack(log_probs, dim=0)) # concat to linear
        values = to_device(torch.stack(values, dim=0).squeeze(-1)) # concat to linear; squeeze(-1), incorrect results due to broadcast

        if self.dones[-1] == True:
            advantages = GAE(self.gamma, self.gae_lambda,None)(self.rewards,values,True,None)
        else:
            fin_next_state = torch.tensor([self.next_states[-1]], device=deviceGPU)
            _, fin_next_state_value = self.actorCritic(fin_next_state)
            advantages = GAE(self.gamma, self.gae_lambda,None)(self.rewards,values,False,fin_next_state_value)
        
        advantages = advantages.detach()


        policy_losses = (-log_probs * advantages) # maximize reward = minimize negative reward
        value_losses = F.mse_loss(values, returns) # mse loss

        loss_weight = 1 # add weight to loss, but here just use 1
        #print(policy_losses)
        #print(value_losses)
        loss = policy_losses.sum() + loss_weight * value_losses.sum() # weighted loss
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.dones[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.saved_actions[:]


def train(lr=0.01):
    hidden_size = 128
    gamma = 0.999
    gae_lambda=0.999
    num_episodes = 10000
    lr_ac = lr

    # Extract the dimensionality of state and action spaces
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]
    
    # Instantiate the agebt and the optimizer
    reinforceAgent = REINFORCEAGENT(observation_dim, action_dim, gamma, gae_lambda, hidden_size, lr_ac)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(reinforceAgent.actorCritic_optim, step_size=100, gamma=0.9)
    
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
            action = reinforceAgent.select_action(state)
            next_state, reward, terminate, truncated ,_ = env.step(action)

            done = terminate
            reinforceAgent.rewards.append(reward)
            reinforceAgent.dones.append(terminate)
            reinforceAgent.next_states.append(next_state)
            ep_reward += reward
            if done or truncated:
                break
            state = torch.tensor([next_state], device=deviceGPU)

        reinforceAgent.actorCritic_optim.zero_grad()   # each time need reset
        loss = reinforceAgent.calculate_loss()
        loss.backward()
        reinforceAgent.actorCritic_optim.step()
        reinforceAgent.clear_memory()    # clean memory
            
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
    plt.savefig('REINFORCE-GAE_plot-whole.png')
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
