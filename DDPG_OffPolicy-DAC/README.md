# DDPG (Deep Deterministic Policy Gradient) Algorithm
## Paper
* DPG: https://proceedings.mlr.press/v32/silver14.html
* DDPG: https://arxiv.org/abs/1509.02971
## Main Algorithm
* A variation of Off-Policy Deterministic Actor-Critic
  * Deep neural networks for function approximation
  * Experience Replaying
  * Ornstein-Uhlenbeck process for exploration
* Actor-Critic
  * Using target network to evaluate next state q value and calculate current q value for MSE loss
  * By DPG theory and chain rule, policy gradient under the deterministic policy is **identical** to the expected gradient of q value under the deterministic policy
    * **Directly using critic Q value as loss function since this value can be derived to DPG formula through chain rule**
    * $`\Large \nabla_{\theta^\mu}J=\frac{1}{N}\Sigma[\nabla_aQ(s,a|\theta^{Q_k})|_{s=s_i,a=\mu(s_i|\theta^\mu)}\nabla_{\theta^\mu}\mu(s|\theta^\mu)|_{s=s_i}]`$, DPG theory
      * $`\Large =\frac{1}{N}\Sigma[\nabla_{\theta^\mu}Q(s,a|\theta^{Q_k})|_{\theta^\mu=\theta^{\mu_k},a=\mu(s_i|\theta^\mu)}]`$, just chain rule
        * $\Large \theta^{\mu_k}$ is current policy
* ![DDPG-flow](ddpgflow.png)
* ![DDPG-structure](ddpg-structure.png)
* ![DDPG-Algorithm](DDPG-algorithm.png)
## Figure Out
* Policy-Based
* Model-Free
* OFF-Policy
* Deterministic Policy
* Actor-Critic
* Per-step training
* Soft copy with $\tau$
* Ornstein-Uhlenbeck process
* CUDA device usage
* Target evaluation without gradient back propagation (add model.eval)
* total_episodes = 300
* batch_size = 128
* gamma      = 0.995
* replay_buffer capacity 100000
* tau = 0.002
* noise_scale = 0.3
* ewma_reward usage
## Environment and Target Game
* gym: 0.26.2
* numpy: 1.26.4 
* pytorch: 2.0.1 
* environment: "Pendulum-v1"
## Result
* ![DDPG](DDPG_plot-whole.png)
