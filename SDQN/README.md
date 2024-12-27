# Soft DQN Algorithm
## Paper
* https://proceedings.mlr.press/v70/haarnoja17a.html
* https://arxiv.org/abs/1912.10891
## Main Algorithm
* Using target network to find maximal q value of next state and evaluate target q value 
  * $y_t = r_{t+1}+\gamma max_{a'}\hat Q(s_{t+1},a',\hat w)$
* ![soft-q-learning-Algorithm](soft_dqn_algorithm.png)
## Figure Out
* Value-Based
* Model-Free
* OFF-Policy
* Per-step training
* Hard copy every 4 step when using target separation
* CUDA device usage
* Target evaluation without gradient back propagation (add model.eval)
* total_episodes = 10000
* batch_size = 16
* gamma      = 0.99
* replay_buffer capacity 50000
* ewma_reward usage
## Environment and Target Game
* gym: 0.26.2
* numpy: 1.26.4 
* pytorch: 2.5.0
* environment: "CartPole-v1"
## Result
* ![Soft DQN](SOFTDQN_plot-whole.png)
