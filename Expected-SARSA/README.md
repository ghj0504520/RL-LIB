# Expected SARSA Algorithm
## Paper
* https://ieeexplore.ieee.org/document/4927542
## Main Algorithm
* $Q(S, A) = Q(S, A) + \alpha(R + (\gamma ExpectedQ(S', A') - Q(S, A))$
  * np.argmax only return one index which is max value and first appeared index
    * even if there are several identical max value
  * Hence, expectation only has one action with greedy policy
* ![EXPSARSA-Algorithm](EXPSARSA-algorithm.png)
## Figure Out
* Value-Based
* Model-Free
* ON-Policy
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
* ![EXPSARSA](EXPSARSA_reward_plot.png)