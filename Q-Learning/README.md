# Q Learning Algorithm
## Paper
* https://link.springer.com/article/10.1007/BF00992698
## Main Algorithm
* $\Large Q(S, A) = Q(S, A) + \alpha(R + (\gamma max_aQ(S', a) - Q(S, A))$
* ![QL-Algorithm](q-learning.png)
## Figure Out
* Value-Based
* Model-Free
* OFF-Policy
* Epsilon greedy
  * epsilon = 0.2
* total_episodes = 20000
* max_steps = 100
* alpha = 0.1
* gamma = 1
  * finite step
## Environment and Target Game
* gym: 0.26.2
* numpy: 1.26.4 
* pytorch: 2.0.1 
* environment: "FrozenLake-v1"
## Result
* ![EXPSARSA](QLearning_reward_plot.png)
