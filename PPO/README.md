# Proximal Policy Optimization (PPO) Algorithm
## Paper
* TRPO
  * https://arxiv.org/abs/1502.05477
  * https://arxiv.org/abs/1909.02769
* PPO
  * https://arxiv.org/abs/1707.06347
  * https://arxiv.org/abs/1906.10306
* PPO Clip
  * https://arxiv.org/abs/2110.13799
  * https://arxiv.org/abs/2312.12065
  * https://doi.org/10.1609/aaai.v38i11.29154
  * https://openreview.net/forum?id=gex-2G2bLdh
* GAE: https://arxiv.org/abs/1506.02438 
## Main Algorithm
* TRPO
  * Trust-Region Policy Optimization
    * Insensitive to learning rate
    * Capable to reach similar results with less data
  * First, the conventional objective of RL
    * $\Large J_\pi(\theta)=V_\pi=\Sigma_a\pi_\theta(a|S)Q_{\pi_\theta}(S,a)$
  * Second, surrogate objective function by Performance Difference Lemma from sampled old policy $\Large \theta_k$
    * $\Large J_\pi(\theta,\theta_k)=\Sigma_a\pi_\theta(a|S)A_{\pi_{\theta_k}}(S,a)$, surrogate function by Performance Difference Lemma (action-state value Q becomes advantage A)
    * $\Large =\Sigma_a\pi_{\theta_k}(a|S)\frac{\pi_{\theta}(a|S)}{\pi_{\theta_k}(a|S)}A_{\pi_{\theta_k}}(S,a)$, importance sampling mechanism
    * where $\Large J_\pi(\theta,\theta_k)$ is surrogate function of $\Large \pi_{\theta}$ under $\Large \pi_{\theta_k}$
* From TRPO to PPO
  * Replace KL constraint with the Lagrangian method by adding penalty terms
  * Dynamically change penalty terms
* From PPO to PPO Clip
  * Just clamp Surrogate Objective (ratio) instead of using KL penalty
* Actor-Critic
  * Deep neural networks for function approximation
  * Sampling action by probability
* TRPO algorithm ![TRPO-Algorithm](trpo.png)
* PPO algorithm, or called PPO KL penalty ![PPO-Algorithm](ppo.png)
  * For continuous, use $\Large \frac{1}{L}\Sigma(log(\pi_{\theta_k}(a|s)) - log(\pi_{\theta}(a|s)))$ to approximate KL divergence of Normal distribution rather than $\Large \Sigma\pi_{\theta_k}(a|s)(log(\pi_{\theta_k}(a|s)) - log(\pi_{\theta}(a|s)))$ 
* PPO Clip algorithm ![PPO-Clip-Algorithm](ppo2.png)
* GAE Estimation for $\Large A_{\pi_{\theta_k}}(S,a)$
  * Figure out handle terminated state and truncated state
* For continuous env, rewards should scale to samller ones.
  * Due to reward scaling, we need to use another evaluation procedure to check performance.
## Figure Out
* Policy-Based
* Model-Free
* ON-Policy
* Actor-Critic
* CUDA device usage
* Discrete:
  learning rate = 0.01
  * total_episodes = 1500
  * gamma = 0.98
  * gae_lambda = 0.95
  * hidden_size = 128
  * kl_beta = 0.1
  * target_kl = 0.01
  * clip_param = 0.2
  * ppo_epoch = 10
  * lr_a=1e-3
  * lr_c=1e-2
* Continuous:
  * total_episodes = 4000
  * gamma = 0.99
  * gae_lambda = 0.95
  * hidden_size = 64
  * kl_beta = 5
  * target_kl = 0.03
  * clip_param = 0.2
  * ppo_epoch = 10
  * entropy_coef = 0.01
  * lr_a=3e-4
  * lr_c=3e-4
* ewma_reward usage
## Environment and Target Game
* gym: 0.26.2
* numpy: 1.26.4 
* pytorch: 2.5.0 
* environment: "CartPole-v1" and "Pendulum-v1"
## Result
* PPO KL penalty for Discrete "CartPole-v1"
  * ![ppo-kl-dis](PPO_KL_Discrete_plot-whole.png)
* PPO Clip for Discrete "CartPole-v1"
  * ![ppo-clip-dis](PPO_CLIP_Discrete_plot-whole.png)
* PPO KL penalty for Continuous "Pendulum-v1"
  * ![ppo-kl-cont](PPO_KL_Continuous_plot-whole.png)
* PPO Clip for Continuous "Pendulum-v1"
  * ![ppo-clip-cont](PPO_CLIP_Continuous_plot-whole.png)
