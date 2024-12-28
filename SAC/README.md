# Soft Actor Critic (SAC) Algorithm
## Paper
* Original Version: https://arxiv.org/abs/1801.01290
  * w/ value network
* Revised Version: https://arxiv.org/abs/1812.05905
  * w/o value network and w/ automatically adapting alpha
* for Discrete: https://arxiv.org/abs/1910.07207
## Main Algorithm
* Maximum long term total reward
  * w/o entropy:
    * Optimal Policy: $`\pi^*_{std}=\arg\max_\pi\Sigma \Bbb E_{(s_t,a_t)}[r(s_t,a_t)]`$
    * Objective: $J(\pi)=\Sigma\Bbb E_{(s_t,a_t)}[r(s_t,a_t)]$
  * w/ entropy:
    * Optimal Policy: $`\pi^*_{entropy}=\arg\max_\pi\Sigma \Bbb E_{(s_t,a_t)}[r(s_t,a_t)+\alpha\mathcal H(\pi(\cdot|s_t))]`$
    * Objective: $J(\pi)=\Sigma\Bbb E_{(s_t,a_t)}[r(s_t,a_t)+\alpha\mathcal H(\pi(\cdot|s_t))]$
    * Entropy: $\mathcal H(P)=\Bbb{E}_{x\sim P}[-\log P(x)]=-\Sigma_xP(x)\log P(x)$
* Soft Policy Improvement:
  * $`\pi_{new} =\arg\min D_{KL}(\pi(\cdot|s_t)||\frac{\exp(Q^{\pi_{old}}_{soft}(s_t,\cdot))}{Z^{\pi_{old}}(s_t)})`$
  * since, $\pi(a_t|s_t)\propto\exp(-\mathcal{E}(s_t,a_t))$ 
    * And $\mathcal{E}(s_t,a_t)=-\frac{1}{\alpha}Q^\pi_{soft}(s_t,a_t)$
    * Then, using softMAX
* Soft Policy Evaluation:
  * Soft Q: 
    * $Q^\pi_{soft}(s_t,a_t)=r_t+\gamma\Bbb E_{s_{t+1}}[V^\pi_{soft}(s_{t+1})]$
  * Soft V: 
    * $V^\pi_{soft}(s_t)=\Bbb E_{a\sim\pi}[Q^\pi_{soft}(s_t,a_t)-\alpha\log\pi(a_t|s_t)]\\ =\alpha\log\Sigma_a\exp(\frac{Q^\pi_{soft}(s_t,a_t)}{\alpha})$
* SAC Objective:
  * $J_Q(\theta)=\Bbb E_{(s_t,a_t)}[\frac{1}{2}(Q^\theta_{soft}(s_t,a_t)-\widehat{Q^{\theta}_{soft}}(s_t,a_t))^2]$
    * $`\widehat{Q^{\theta}_{soft}}(s_t,a_t)=r_t+\gamma\Bbb E_{s_{t+1}}[{V^{\bar\psi}_{soft}}(s_{t+1})]`$
    * Using SGD, $`\widehat{Q^{\theta}_{soft}}(s_t,a_t)\simeq r_t+\gamma {V^{\bar\psi}_{soft}}(s_{t+1})`$
    * Advanced version:
      * $`\widehat{Q^{\theta}_{soft}}(s_t,a_t)=r_t+\gamma\Bbb E_{a\sim\pi}[\min_{1,2}Q^{\bar\theta}_{soft}(s_t,a_t)-\alpha\log\pi_\phi(a_t|s_t)]`$
      * SGD, $`\widehat{Q^{\theta}_{soft}}(s_t,a_t)\simeq r_t+\gamma(\min_{1,2}Q^{\bar\theta}_{soft}(s_t,a_t)-\alpha\log\pi_\phi(a_t|s_t))`$
  * $J_V(\psi)=\Bbb E_{s_t}[\frac{1}{2}(V^\psi_{soft}(s_t)-\widehat{V^{\psi}_{soft}}(s_t))^2]$
    * $`\widehat{V^{\psi}_{soft}}(s_t)=\Bbb E_{a\sim\pi}[\min_{1,2}Q^\theta_{soft}(s_t,a_t)-\alpha\log\pi_\phi(a_t|s_t)]`$
    * Using SGD, $`\widehat{V^{\psi}_{soft}}(s_t)\simeq \min_{1,2}Q^\theta_{soft}(s_t,a_t)-\alpha\log\pi_\phi(a_t|s_t)`$
  * $`J_\pi(\phi)=(\min)\Bbb{E_{a_t,s_t}}[D_{KL}(\pi_\phi(\cdot|s_t)||\frac{\exp(Q^\theta_{soft}(s_t,\cdot))}{Z_\theta(s_t)})] \\=(\min)\Bbb E[\alpha\log\pi_\phi(a_t|s_t)-\min_{1,2}Q^\theta_{soft}(s_t,a_t)]`$
    * Also equal to $`(\max)\Bbb E[\min_{1,2}Q^\theta_{soft}(s_t,a_t)-\alpha\log\pi_\phi(a_t|s_t)]\\=(\max)\Bbb E[\Sigma_t r(s_t,a_t)+\alpha\mathcal H(\pi(\cdot|s_t))]`$
      * Maximum long term total reward with entropy
    * $a_t=f_\phi(\epsilon_t;s_t)$, $\epsilon$ is noise
      * Reparameterization trick
    * Objective is  minimizing the KL-divergence
  * $J(\alpha)=\Bbb{E}[-\alpha\log\pi_t(a_t|s_t)-\alpha\bar {\mathcal H}]$
* Original version SAC:
  * ![SAC-Algorithm](sac-algorithm.png)
  * Value net, target Value net, 2 Q net (TD3), Policy net
* Advanced and Revised version SAC:
  * ![adv-SAC-Algorithm](adv-sac-algorithm.png)
  * 2 Q net, 2 target Q net, Policy net
## Figure Out
* Policy-Based
* Model-Free
* OFF-Policy
* Per-step training
* Soft copy every step
* CUDA device usage
* Target evaluation without gradient back propagation (add model.eval)
* total_episodes = 500
* batch_size = 128(original ver)/300(revised ver)
* gamma      = 0.99
* soft_tau   = 1e-2
* value learning rate = 3e-4
* soft_q learning rate = 3e-4
* policy learning rate = 3e-4
* alpha learning rate = 3e-4
* replay_buffer capacity 1000000
* ewma_reward usage
## Environment and Target Game
* gym: 0.26.2
* numpy: 1.26.4 
* pytorch: 2.5.0
* environment: "Pendulum-v1"
## Result
* Original SAC with value network
  * ![SAC-original](SAC_plot-whole.png)
* Revised and advanced SAC without value network and with automatically adapting alpha
  * ![SAC-revised](ADV_SAC_plot-whole.png)
