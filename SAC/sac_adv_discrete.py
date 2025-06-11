import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import namedtuple


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


# Replay buffer utility
Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) #state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)


# SAC Q net for discrete
class SoftQNetwork(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_actions):
        super(SoftQNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),  # Replace with the provided activation if needed
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, inputs):
        x = self.layers(inputs)
        return x
        

# SAC policy net for discrete
class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_actions):
        super(PolicyNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),  # Replace with the provided activation if needed
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        probs = self.layers(state)
        return probs
    
    def evaluate(self, state, epsilon=1e-8):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        probs = self.forward(state)
        log_probs = torch.log(probs)

        # stable result
        z = (probs == 0.0).float() * epsilon
        log_probs = torch.log(probs + z)

        return log_probs


# Advanced SOFT ACTOR and CRITIC AGENT for discrete
class ADVSOFTACTORCRITICAGENT():
    def __init__(self, hidden_dim, q_lr = 3e-4, p_lr = 3e-4, a_lr = 3e-4):

        self.soft_q_net1 = SoftQNetwork(hidden_dim, state_dim, action_dim).to(deviceGPU)
        self.soft_q_net2 = SoftQNetwork(hidden_dim, state_dim, action_dim).to(deviceGPU)
        self.target_soft_q_net1 = SoftQNetwork(hidden_dim, state_dim, action_dim).to(deviceGPU)
        self.target_soft_q_net2 = SoftQNetwork(hidden_dim, state_dim, action_dim).to(deviceGPU)
        hard_update(self.target_soft_q_net1, self.soft_q_net1)
        hard_update(self.target_soft_q_net2, self.soft_q_net2)

        self.policy_net = PolicyNetwork(hidden_dim, state_dim, action_dim).to(deviceGPU)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=deviceGPU)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=p_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=a_lr)
    
    def action_selection(self, state, deterministic):
        probs = self.policy_net(state)
        dist = Categorical(probs)

        if deterministic:
            action = np.argmax(probs.detach().cpu().numpy())
        else:
            action = dist.sample().squeeze().detach().cpu().numpy()
        return action.item()
        
    def update(self, batch, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        state = to_device(torch.cat([b.state for b in batch]))
        action = to_device(torch.cat([b.action for b in batch]))
        reward = to_device(torch.cat([b.reward for b in batch]).unsqueeze(1))
        done = to_device(torch.cat([b.mask for b in batch]).unsqueeze(1))
        next_state = to_device(torch.cat([b.next_state for b in batch]))

        '''
        print(state.shape)
        print(next_state.shape)
        print(action.shape)
        print(reward.shape)
        print(done.shape)
        '''

        predicted_q_value1 = self.soft_q_net1(state)
        predicted_q_value1 = predicted_q_value1.gather(1, action.unsqueeze(-1))
        predicted_q_value2 = self.soft_q_net2(state)
        predicted_q_value2 = predicted_q_value2.gather(1, action.unsqueeze(-1))

        log_prob = self.policy_net.evaluate(state)
        
        with torch.no_grad():
            next_log_prob = self.policy_net.evaluate(next_state)
        
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            policy_probs = log_prob.exp()
            entropy = (policy_probs * log_prob).sum(dim=1)
            alpha_loss = -(self.log_alpha * (entropy + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        target_q_min = (next_log_prob.exp() * (torch.min(self.target_soft_q_net1(next_state),self.target_soft_q_net2(next_state)) - self.alpha * next_log_prob)).sum(dim=-1).unsqueeze(-1)
        target_q_value = reward + (1 - done) * gamma * target_q_min 
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  

        # Training Policy Function
        with torch.no_grad(): # without reparameterization action, q value not dependent on policy
            predicted_new_q_value = torch.min(self.soft_q_net1(state),self.soft_q_net2(state))
        policy_loss = (log_prob.exp()*(self.alpha * log_prob - predicted_new_q_value)).sum(dim=-1).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update the target value net
        soft_update(self.target_soft_q_net1, self.soft_q_net1, soft_tau)
        soft_update(self.target_soft_q_net2, self.soft_q_net2, soft_tau)
            
        return policy_loss, q_value_loss1+q_value_loss2


# MAIN
env = gym.make('CartPole-v1')
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n  # discrete

env.reset(seed=1)
np.random.seed(1)
random.seed(1)
torch.manual_seed(1) # config for CPU
torch.cuda.manual_seed(1) # config for GPU

# hyper-parameters for RL training
max_episodes  = 500
batch_size  = 256
AUTO_ENTROPY=True
DETERMINISTIC=False
hidden_dim = 512
replay_buffer_size = 1e6
#target_entropy = -1.*action_dim
# use 0.6 instead of suggested 0.98 for better effect
target_entropy = 0.6 * -np.log(1 / action_dim)
q_losses = []
p_losses = []
all_rewards = []
ewma_reward = 0

replay_buffer = ReplayMemory(replay_buffer_size)
advSACDAgent=ADVSOFTACTORCRITICAGENT(hidden_dim=hidden_dim)

if __name__ == '__main__':

    for episode_idx in range(max_episodes):
        initState, _ =  env.reset()
        state = torch.Tensor([initState]).to(deviceGPU)

        episode_q_loss = 0.0
        episode_p_loss = 0.0
        episode_reward = 0.0
        step = 0

        terminate = False
        truncated = False

        while not terminate and not truncated:
            step = step+1

            action = advSACDAgent.action_selection(state, deterministic = DETERMINISTIC)

            next_state, reward, terminate, truncated, _ = env.step(action)
            # env.render()       
                
            #replay_buffer.push(state, action, reward, next_state, terminate)
            replay_buffer.push(
                state, 
                torch.tensor([action], device=deviceGPU), 
                torch.tensor([terminate], dtype=torch.float32, device=deviceGPU),
                torch.tensor([next_state], device=deviceGPU),
                torch.tensor([reward], dtype=torch.float32, device=deviceGPU)
                )
            
            state = torch.tensor([next_state], device=deviceGPU)
            episode_reward += reward
            
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                ploss, qloss=advSACDAgent.update(batch, reward_scale=1., auto_entropy=AUTO_ENTROPY, target_entropy=target_entropy)
                episode_p_loss = episode_p_loss+ploss.item()
                episode_q_loss = episode_q_loss + qloss.item()*0.5
            if terminate:
                break

        q_losses.append(episode_q_loss/step)
        p_losses.append(episode_p_loss/step)
        ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
        all_rewards.append(ewma_reward)

        if episode_idx % 1 == 0:
            print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}\t p_loss: {}, q_loss: {}'.format(episode_idx, step, 
                                                    episode_reward, ewma_reward, episode_p_loss/step, episode_q_loss/step))


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
    plt.plot(q_losses, label='Value Loss', color='orange')
    plt.title('Value Loss Over Episodes')
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot EWMA reward
    plt.subplot(1, 3, 3)
    plt.plot(all_rewards, label='EWMA Reward', color='green')
    plt.title('EWMA Reward Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('EWMA Reward')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save the plot as an image
    plt.savefig('ADV_SAC_Discrete_plot-whole.png')
    plt.show()


