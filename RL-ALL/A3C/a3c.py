import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import numpy as np

from queue import Empty
from collections import namedtuple
import matplotlib.pyplot as plt

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value','entropy'])


# Training Strategy
GLOBAL_MAX_EPISODE = 1200
GAMMA = 0.999
CPUNUM = 10


# Copy Network
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# Multi Process Share Optim Info between Processes
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1) # figure out
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


# NN Model
class PolicyNet(torch.nn.Module):
    def __init__(self, hidden_dim, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        x = self.layers(state)
        return x

class ValueNet(torch.nn.Module):
    def __init__(self, hidden_dim, state_dim):
        super(ValueNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        x = self.layers(state)
        return x


# A3CAGENT
class A3CAGENT:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, env, numOfCPU):
        # global network
        self.global_actor = PolicyNet(hidden_dim, state_dim, action_dim)
        self.global_critic = ValueNet(hidden_dim, state_dim)

        # share the global parameters in multiprocessing
        self.global_actor.share_memory()
        self.global_critic.share_memory()
        self.global_critic_optimizer = SharedAdam(self.global_critic.parameters(), lr=critic_lr, betas=(0.92, 0.999))
        self.global_actor_optimizer = SharedAdam(self.global_actor.parameters(), lr=actor_lr, betas=(0.92, 0.999))
        
        self.env = env
        self.global_episode = mp.Value('i', 0)
        self.global_episode_reward = mp.Value('d', 0.)
        self.res_queue = mp.Queue()
        
        # worker
        self.workers = [Worker(i, self.global_actor, self.global_critic, self.global_critic_optimizer,
                          self.global_actor_optimizer,  self.env, state_dim, hidden_dim, action_dim,
                          self.global_episode, self.global_episode_reward, self.res_queue) for i in range(numOfCPU)]
    
    def optimal_action(self, state): # for evaluation
        logits = self.global_actor(state.squeeze())
        dist = F.softmax(logits, dim=0)
        action = torch.argmax(dist).item()        
        return action
    
    def train(self):
        [w.start() for w in self.workers]
        res = []
        while True:
            r = self.res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
        [w.join() for w in self.workers]


# Child Process
class Worker(mp.Process):
    def __init__(self, name, global_actor, global_critic, global_critic_optimizer, global_actor_optimizer,
                 env, state_dim, hidden_dim, action_dim, global_episode, global_episode_reward, res_queue):
        super(Worker, self).__init__()
        self.id = name
        self.name = 'w%02i' % name
        self.env = env

        self.global_episode = global_episode
        self.global_episode_reward = global_episode_reward
        self.res_queue = res_queue
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.global_critic_optimizer = global_critic_optimizer
        self.global_actor_optimizer = global_actor_optimizer

        self.local_actor = PolicyNet(hidden_dim, state_dim, action_dim)
        self.local_critic = ValueNet(hidden_dim, state_dim)

        self.saved_actions = []
        self.rewards = []

    # stochastic sampling
    def take_action(self, state):
        logits = self.local_actor(state.squeeze())
        dist = F.softmax(logits, dim=0)
        probs = torch.distributions.Categorical(dist)
        state_value = self.local_critic(state.squeeze())
        action = probs.sample()
        
        # save to action buffer
        self.saved_actions.append(SavedAction(probs.log_prob(action), 
                                              state_value, probs.entropy()))
        
        return action.detach().item()

    def update(self):

        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []
        
        with torch.no_grad():
            for t in reversed(range(len(self.rewards))):
                R = self.rewards[t] + R * GAMMA
                returns.insert(0, R)
            # Convert returns to a tensor and move to the appropriate device
            returns = (torch.tensor(returns).unsqueeze(1))

        log_probs = [action.log_prob for action in saved_actions] # get log probs list from saved_actions
        values = [action.value for action  in saved_actions] # get values list from saved_actions
        entropies = [action.entropy for action  in saved_actions]

        log_probs = (torch.stack(log_probs, dim=0)) # concat to linear
        values = (torch.stack(values, dim=0)) # concat to linear
        entropies = (torch.stack(entropies, dim=0))

        advantage = (returns-values).detach()

        policy_losses = (-log_probs * advantage) # maximize reward = minimize negative reward
        critic_loss = F.mse_loss(values, returns) # mse loss
        entropy_losses = - entropies
        actor_loss = policy_losses.mean() + entropy_losses.mean() * 0.001

        self.global_critic_optimizer.zero_grad()
        critic_loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.local_critic.parameters(), self.global_critic.parameters()):
            global_params._grad = local_params._grad
        self.global_critic_optimizer.step()


        self.global_actor_optimizer.zero_grad()
        actor_loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.local_actor.parameters(), self.global_actor.parameters()):
            global_params._grad = local_params._grad
        self.global_actor_optimizer.step()

        self.clear_memory()

    def sync_with_global(self):
        hard_update(self.local_critic,self.global_critic)
        hard_update(self.local_actor,self.global_actor)
        
    def run(self):
        while self.global_episode.value < GLOBAL_MAX_EPISODE:
            initState, _ = self.env.reset(seed=self.id)
            state = torch.tensor([initState], dtype=torch.float32)

            terminated = False
            truncated = False
            ep_reward = 0
            
            while self.global_episode.value < GLOBAL_MAX_EPISODE:

                action = self.take_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                self.rewards.append(reward)
                ep_reward += reward

                if terminated or truncated:
                    with self.global_episode.get_lock():
                        self.global_episode.value += 1
                    print(self.name + " | episode: " + str(self.global_episode.value) + " " + str(ep_reward))
                    self.res_queue.put(ep_reward)
                    self.update()
                    self.sync_with_global()
                    break
                
                state = torch.tensor([next_state], dtype=torch.float32)
    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]



# MAIN
def main_test():
    env = gym.make("CartPole-v1")
    randseed = 9
    env.reset(seed=randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed) # config for CPU
    torch.cuda.manual_seed(randseed) # config for GPU

    a3cAgent = A3CAGENT(state_dim=env.observation_space.shape[0], hidden_dim=256, action_dim=env.action_space.n,
                     actor_lr=1e-3, critic_lr=1e-3, env=env, numOfCPU=CPUNUM)
    [w.start() for w in a3cAgent.workers]
    returnLists1 = []
    while True:
        try:
            r = a3cAgent.res_queue.get(timeout=10)  # 3sec timeout
            if r is not None:
                returnLists1.append(r)
            else:
                break
        except Empty:  # 当队列中没有数据可供获取时，get方法会抛出Empty异常
            print("No data in queue, breaking...")
            break
    [w.join() for w in a3cAgent.workers]
    


    ewma_rewards = []
    ewma_reward = 0  # Initialize EWMA reward
    
    for reward in returnLists1:
        ewma_reward = 0.05 * reward + (1 - 0.05) * ewma_reward
        ewma_rewards.append(ewma_reward)
    # Plot the losses and EWMA reward after training
    plt.figure(figsize=(12, 6))
    # Plot Ep reward
    plt.subplot(1, 2, 1)
    plt.plot(returnLists1, label='Episode Reward', color='blue')
    plt.title('Episode Reward Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.legend()
    # Plot EWMA reward
    plt.subplot(1, 2, 2)
    plt.plot(ewma_rewards, label='EWMA Reward', color='green')
    plt.title('EWMA Reward Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('EWMA Reward')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    # Save the plot as an image
    plt.savefig('A3C_plot-whole-2.png')
    #plt.show()



    a3cAgent.global_actor.eval()
    with torch.no_grad():
        for eps in range(10):
            initState, _ = env.reset()
            state = torch.tensor([initState], dtype=torch.float32)
            episode_reward = 0

            terminate = False
            truncated = False   

            while not terminate and not truncated:
                action = a3cAgent.optimal_action(state)
                next_state, reward, terminate, truncated, _ = env.step(action)

                episode_reward += reward
                state = torch.tensor([next_state], dtype=torch.float32)
            print('Test Episode: ', eps, '| Episode Reward: ', episode_reward)

    env.close()

main_test()