# Quantile Regression DQN (QRDQN) Algorithm
## Paper
* https://arxiv.org/abs/1710.10044
## Main Algorithm
* 
* ![QRDQN-Algorithm](QRDQN.png)
## Figure Out
* Distributional Value-Based
* Model-Free
* OFF-Policy
* Per-step training
* Hard copy every 100 step
* Epsilon greedy decay as episodes increase
* CUDA device usage
* Target evaluation without gradient back propagation (add model.eval)
* total_episodes = 10000 with early terminating
* batch_size = 64
* gamma      = 0.99
* replay_buffer capacity 10000
* num_support Quantile = 8
* ewma_reward usage
## Environment and Target Game
* gym: 0.26.2
* numpy: 1.26.4 
* pytorch: 2.0.1 
* environment: "CartPole-v1"
## Result
* ![QRDQN](QRDQN_plot-whole.png)
