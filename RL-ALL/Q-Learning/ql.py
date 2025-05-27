import numpy as np
import gym
import matplotlib.pyplot as plt

class QLAgent:
    """
    The Agent that uses Q Learning update to improve it's behaviour
    """
    def __init__(self, epsilon, alpha, gamma, num_state, num_actions, action_space):
        """
        Constructor
        Args:
            epsilon: The degree of exploration
            gamma: The discount factor
            num_state: The number of states
            num_actions: The number of actions
            action_space: To call the random action
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_state = num_state
        self.num_actions = num_actions
 
        #Initializing the Q-matrix
        self.Q = np.zeros((self.num_state, self.num_actions))
        self.action_space = action_space

    #Function to choose the next action ε-Greedy
    def choose_action(self, state):
        action=0
        if np.random.uniform(0, 1) < self.epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])
        return action

    #Function to learn the Q-value
    def update(self, prev_state, prev_action, reward, next_state, next_action): 
        """
        Update the action value function using the Q Learning update.
        Q(S, A) = Q(S, A) + alpha(reward + (gamma * maxQ(S_, A_) - Q(S, A))
        Args:
            prev_state: The previous state
            next_state: The next state
            reward: The reward for taking the respective action
            prev_action: The previous action
            next_action: The next action
        Returns:
            None
        """
        predict = self.Q[prev_state, prev_action]
        target = reward + self.gamma * np.max(self.Q[next_state, :])
        self.Q[prev_state, prev_action] += self.alpha * (target - predict)



#Building the environment
env = gym.make('FrozenLake-v1')


#Defining the different parameters
epsilon = 0.2 #0.9
total_episodes = 10000 #10000
max_steps = 100
alpha = 0.1 #0.85
gamma = 1 #0.95


qlAgent = QLAgent(
    epsilon, alpha, gamma, env.observation_space.n, 
    env.action_space.n, env.action_space)

totalReward = []

# Starting the Q learning
for episode in range(total_episodes):
    t = 0
    state1, info = env.reset()
    action1 = qlAgent.choose_action(state1)
    terminate = False
    truncated = False
    #Initializing the reward
    episodesReward = 0
    while not terminate and not truncated:
        #Visualizing the training
        env.render()

        #Getting the next state
        state2, reward, terminate, truncated, info = env.step(action1)

        #Choosing the next action
        action2 = qlAgent.choose_action(state2)
         
        #Learning the Q-value
        qlAgent.update(state1, action1, reward, state2, action2)
 
        state1 = state2
        action1 = action2
         
        #Updating the respective vaLues
        t += 1
        episodesReward += reward
         
        #If at the end of learning process
        if terminate:
            break
    totalReward.append(episodesReward)

env.close()
#Evaluating the performance
print ("Performance : ", np.mean(totalReward))
 
#Visualizing the Q-matrix
print(qlAgent.Q)


chunk_size = 100  # Every 100 episodes
num_chunks = len(totalReward) // chunk_size  # Number of chunks

# Calculate average reward for every 1000 episodes
average_rewards = [sum(totalReward[i * chunk_size:(i + 1) * chunk_size]) / chunk_size for i in range(num_chunks)]

# Plot the average rewards for every 1000 episodes
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_chunks + 1), average_rewards, label='Q Learning AVG Reward per 100 Episodes', color='b', marker='o')



# Add labels and title
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Reward Tendency Over Episodes')

# Show legend
plt.legend()

# Show grid
plt.grid(True)

# Save the plot as an image
plt.savefig('QLearning_reward_plot.png')

# Display the plot
# plt.show()
