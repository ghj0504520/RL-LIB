import gym
import random
import collections
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
#=====
# single thread version; no trust-region updates
#=====

# Define a useful tuple (optional)
Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'reward', 'log_prob'))

class ReplayBuffer():
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)

    def put(self, seq_data):
        self.memory.append(seq_data)
    
    def sample(self, on_policy=False):
        if on_policy:
            mini_batch = [self.memory[-1]]
        else:
            mini_batch = random.sample(self.memory, batch_size)
        return mini_batch 

    def size(self):
        return len(self.memory)

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
        self.commonLayer = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU()            
        )
        self.actorLayer = nn.Sequential(
            nn.Linear(hidden_size, num_outputs)
        )
        self.valueLayer = nn.Sequential(
            nn.Linear(hidden_size, num_outputs)
        )
        
    def policy(self, x, softmax_dim = 0):
        x = self.commonLayer(x)
        x = self.actorLayer(x)
        pi = F.softmax(x, dim=softmax_dim)
        return pi
    
    def actionStateV(self, x):
        x = self.commonLayer(x)
        q = self.valueLayer(x)
        return q


class ACERAGENT(object):
    def __init__(self, num_inputs, action_dim, gamma=0.999, hidden_size=256, lr_ac=0.01):
        super(ACERAGENT, self).__init__()
        self.num_inputs = num_inputs
        self.action_dim = action_dim        
        self.gamma = gamma

        self.actorCritic = ActorCritic(hidden_size, num_inputs, action_dim).to(deviceGPU)
        self.actorCritic_optim = optim.Adam(self.actorCritic.parameters(), lr=lr_ac)

    def select_action(self, state):
            # cannot disable gradient
            prob = self.actorCritic.policy(state.squeeze())
            #print(state.squeeze(1).shape)
            action_dis = Categorical(prob)        
            action = action_dis.sample()
            
            return prob, action.item()

    def update_parameter(self, batch):
        state_lst = action_lst = reward_lst = log_prob_lst = done_lst = is_first_lst = None
        for seq in batch:
            state = torch.cat([b.state for b in seq])
            action = torch.cat([b.action for b in seq])
            reward = torch.cat([b.reward for b in seq])
            log_prob = torch.cat([b.log_prob for b in seq])
            done = torch.cat([1 - b.mask for b in seq])
            #is_first = True  # Flag for indicating whether the transition is the first item from a sequence
            first_mask = torch.zeros(len(seq), dtype=torch.bool)
            first_mask[0] = True

            if state_lst is None:
                state_lst, action_lst, reward_lst, log_prob_lst, done_lst, is_first_lst = \
                    state, action, reward, log_prob, done, first_mask
            else:
                state_lst = torch.cat([state_lst, state])
                action_lst = torch.cat([action_lst, action])
                reward_lst = torch.cat([reward_lst, reward])
                log_prob_lst = torch.cat([log_prob_lst, log_prob])
                done_lst = torch.cat([done_lst, done])
                is_first_lst= torch.cat([is_first_lst, first_mask])

        state_batch,action_batch,reward_batch,log_prob_batch,done_batch,is_first_batch = \
            state_lst, action_lst.unsqueeze(1), reward_lst, log_prob_lst, done_lst, is_first_lst

        '''
        print(s.shape)
        print(a.shape)
        print(r.shape)
        print(prob.shape)
        print(done_mask.shape)
        print(is_first.shape)
        '''
        q_value = self.actorCritic.actionStateV(state_batch)
        #print(q.shape)
        q_action_value = q_value.gather(1,action_batch)
        policy = self.actorCritic.policy(state_batch.squeeze(1), softmax_dim = 1)
        #print(pi.shape)
        action_policy = policy.gather(1,action_batch)
        state_value = (q_value * policy).sum(1).unsqueeze(1).detach()
        
        # bias terms
        rho = policy.detach()/(log_prob_batch)
        #print(rho.shape)
        rho_a = rho.gather(1,action_batch)
        #print(rho_a.shape)
        rho_bar = rho_a.clamp(max=c)
        correction_coeff = (1-c/rho).clamp(min=0)

        # retrace calculation
        q_retrace = state_value[-1] * done_batch[-1]
        q_retrace_lst = []
        for i in reversed(range(len(reward_batch))):
            q_retrace = reward_batch[i] + self.gamma * q_retrace
            q_retrace_lst.append(q_retrace.item())
            q_retrace = rho_bar[i] * (q_retrace - q_action_value[i]) + state_value[i]
            
            if is_first_batch[i] and i!=0:
                q_retrace = state_value[i-1] * done_batch[i-1] # When a new sequence begins, q_ret is initialized  
                
        q_retrace_lst.reverse()
        q_retrace = torch.tensor(q_retrace_lst, dtype=torch.float, device=deviceGPU).unsqueeze(1)
        
        loss1 = -rho_bar * torch.log(action_policy) * (q_retrace - state_value)  # important sampling and retrace term
        loss2 = -correction_coeff * policy.detach() * torch.log(policy) * (q_value.detach()-state_value) # bias correction term
        loss = loss1 + loss2.sum(1) + F.smooth_l1_loss(q_action_value, q_retrace)
        
        self.actorCritic_optim.zero_grad()
        loss.mean().backward()
        self.actorCritic_optim.step()

        return loss.mean()


#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98
buffer_limit  = 6000  
rollout_len   = 10    # rollouts to limit variance 
batch_size    = 4     # Indicates 4 sequences per mini-batch (4*rollout_len = 40 samples total)
c             = 1.0   # For truncating importance sampling ratio

def main():
    env = gym.make('CartPole-v1')
    rSeed = 2
    env.action_space.seed(rSeed)
    env.reset(seed=rSeed)
    random.seed(rSeed)
    torch.manual_seed(rSeed) # config for CPU
    torch.cuda.manual_seed(rSeed) # config for GPU

    replay_buffer = ReplayBuffer(buffer_limit)
    acerAgent = ACERAGENT(env.observation_space.shape[0],env.action_space.n, gamma, 256, learning_rate)

    ewma_reward = 0
    loss = 0
    losses = []
    all_rewards = []

    for episode_idx in range(1700):
        initState, _ = env.reset()
        state = torch.tensor([initState], device=deviceGPU)
        terminate = False
        truncated = False

        episode_reward = 0
        step =0
        loss = 0

        while not terminate and not truncated:
            trajectory = []
            
            for t in range(rollout_len):
                step+=1
                prob, action = acerAgent.select_action(state)
                next_state, reward, terminate, truncated, _ = env.step(action)
                trajectory.append(Transition(
                    state, 
                    torch.tensor([action], device=deviceGPU), 
                    torch.tensor([terminate], dtype=torch.float32, device=deviceGPU), 
                    torch.tensor([reward/100.0], device=deviceGPU), 
                    torch.tensor([prob.detach().cpu().numpy()], device=deviceGPU)
                    )) # prevents unstable Q-values, reduces variance in importance sampling

                episode_reward +=reward
                state = torch.tensor([next_state], device=deviceGPU)
                if terminate:
                    break
                    
            replay_buffer.put(trajectory)
            if replay_buffer.size()>500:
                training_batch = replay_buffer.sample(on_policy=True)
                tmpLoss = acerAgent.update_parameter(training_batch)
                loss = tmpLoss.item()
                
                training_batch = replay_buffer.sample(on_policy=False)
                tmpLoss = acerAgent.update_parameter(training_batch)
                loss += tmpLoss.item()
        
        losses.append(loss*0.5/step)

        ewma_reward =  (1 - 0.05) * ewma_reward + 0.05 * episode_reward
        
        all_rewards.append(ewma_reward)

        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}\t loss: {}'.format(episode_idx, step, 
                                                                                       episode_reward, ewma_reward, loss*0.5/step))

    env.close()

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
    plt.savefig('ACER_plot-whole.png')
    plt.show()

if __name__ == '__main__':
    main()