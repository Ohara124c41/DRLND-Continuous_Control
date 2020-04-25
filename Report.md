# Deep Q-Learning Network (DQN) Agent Continuous Control Project

### Introduction



[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


For this project, the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment was used.

![Trained Agent][image1]

## Environment Details

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

This project was completed using the Udacity Workspace with GPU processing for a single agent. [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents) is used at the baseline for creating the environment. There is a second version which contains 20 identical agents, each with its own copy of the environment. The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience and will be explored at a later time.


The task is episodic, and **in order to solve the environment, agent must get an average score of +30 over 100 consecutive episodes.** This Report.md describes an off-policy Deep Deterministic Policy Gradient (DDPG) implementation.


## Agent Implementation

### Deep Deterministic Policy Gradient (DDPG)

[Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) builds upon Deep Q-Learning Networks by simultaneoulsy learning a policy and a Q-function via the Bellman equation. The content below is referenced from [OpenAI](https://spinningup.openai.com/en/latest/algorithms/ddpg.html).

This approach is closely connected to Q-learning, and is motivated the same way: if you know the optimal action-value function :math:`Q^*(s,a)`, then in any given state, the optimal action :math:`a^*(s)` can be found by solving

.. math::

    a^*(s) = \arg \max_a Q^*(s,a).

DDPG interleaves learning an approximator to :math:`Q^*(s,a)` with learning an approximator to :math:`a^*(s)`, and it does so in a way which is specifically adapted for environments with continuous action spaces by relating to how  the max over actions in :math:`\max_a Q^*(s,a)` are computed.

As the action space is continuous, the function :math:`Q^*(s,a)` is presumed to be differentiable with respect to the action argument. This allows for an efficient, gradient-based learning rule for a policy :math:`\mu(s)` which exploits that fact. Afterward, and approximation with :math:`\max_a Q(s,a) \approx Q(s,\mu(s))` can be derived.

#### Replay Buffers.

All standard algorithms for training a deep neural network to approximate :math:`Q^*(s,a)` make use of an experience replay buffer. This is the set :math:`{\mathcal D}` of previous experiences. In order for the algorithm to have stable behavior, the replay buffer should be large enough to contain a wide range of experiences, but it may not always be good to keep everything. If you only use the very-most recent data, you will overfit to that and things will break; if you use too much experience, you may slow down your learning. This may take some tuning to get right.

#### Target Networks

Q-learning algorithms make use of **target networks**. The term

.. math::

    r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a')

is called the **target**, because when we minimize the MSBE loss, we are trying to make the Q-function be more like this target. Problematically, the target depends on the same parameters we are trying to train: :math:`\phi`. This makes MSBE minimization unstable. The solution is to use a set of parameters which comes close to :math:`\phi`, but with a time delay---that is to say, a second network, called the target network, which lags the first. The parameters of the target network are denoted :math:`\phi_{\text{targ}}`.

In DQN-based algorithms, the target network is just copied over from the main network every some-fixed-number of steps. In DDPG-style algorithms, the target network is updated once per main network update by polyak averaging:

.. math::

    \phi_{\text{targ}} \leftarrow \rho \phi_{\text{targ}} + (1 - \rho) \phi,

where :math:`\rho` is a hyperparameter between 0 and 1 (usually close to 1). (This hyperparameter is called ``polyak`` in our code).

#### DDPG Specific

Computing the maximum over actions in the target is a challenge in continuous action spaces. DDPG deals with this by using a **target policy network** to compute an action which approximately maximizes :math:`Q_{\phi_{\text{targ}}}`. The target policy network is found the same way as the target Q-function: by polyak averaging the policy parameters over the course of training.

Putting it all together, Q-learning in DDPG is performed by minimizing the following MSBE loss with stochastic gradient descent:

.. math::

    L(\phi, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
        \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) Q_{\phi_{\text{targ}}}(s', \mu_{\theta_{\text{targ}}}(s')) \right) \Bigg)^2
        \right],

where :math:`\mu_{\theta_{\text{targ}}}` is the target policy.


#### The Policy Learning Side of DDPG

Policy learning in DDPG is fairly simple. We want to learn a deterministic policy :math:`\mu_{\theta}(s)` which gives the action that maximizes :math:`Q_{\phi}(s,a)`. Because the action space is continuous, and we assume the Q-function is differentiable with respect to action, we can just perform gradient ascent (with respect to policy parameters only) to solve

.. math::

    \max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi}(s, \mu_{\theta}(s)) \right].

Note that the Q-function parameters are treated as constants here.

Below, the pseudocode is described:

#### Pseudocode
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Deep Deterministic Policy Gradient}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta$, Q-function parameters $\phi$, empty replay buffer $\mathcal{D}$
        \STATE Set target parameters equal to main parameters $\theta_{\text{targ}} \leftarrow \theta$, $\phi_{\text{targ}} \leftarrow \phi$
        \REPEAT
            \STATE Observe state $s$ and select action $a = \text{clip}(\mu_{\theta}(s) + \epsilon, a_{Low}, a_{High})$, where $\epsilon \sim \mathcal{N}$
            \STATE Execute $a$ in the environment
            \STATE Observe next state $s'$, reward $r$, and done signal $d$ to indicate whether $s'$ is terminal
            \STATE Store $(s,a,r,s',d)$ in replay buffer $\mathcal{D}$
            \STATE If $s'$ is terminal, reset environment state.
            \IF{it's time to update}
                \FOR{however many updates}
                    \STATE Randomly sample a batch of transitions, $B = \{ (s,a,r,s',d) \}$ from $\mathcal{D}$
                    \STATE Compute targets
                    \begin{equation*}
                        y(r,s',d) = r + \gamma (1-d) Q_{\phi_{\text{targ}}}(s', \mu_{\theta_{\text{targ}}}(s'))
                    \end{equation*}
                    \STATE Update Q-function by one step of gradient descent using
                    \begin{equation*}
                        \nabla_{\phi} \frac{1}{|B|}\sum_{(s,a,r,s',d) \in B} \left( Q_{\phi}(s,a) - y(r,s',d) \right)^2
                    \end{equation*}
                    \STATE Update policy by one step of gradient ascent using
                    \begin{equation*}
                        \nabla_{\theta} \frac{1}{|B|}\sum_{s \in B}Q_{\phi}(s, \mu_{\theta}(s))
                    \end{equation*}
                    \STATE Update target networks with
                    \begin{align*}
                        \phi_{\text{targ}} &\leftarrow \rho \phi_{\text{targ}} + (1-\rho) \phi \\
                        \theta_{\text{targ}} &\leftarrow \rho \theta_{\text{targ}} + (1-\rho) \theta
                    \end{align*}
                \ENDFOR
            \ENDIF
        \UNTIL{convergence}
    \end{algorithmic}
    \end{algorithm}


## Model
The next two entries visualize the flow diagrams for the Network. This work builds upon implementations from the first DRLND project [Navigation](https://github.com/Ohara124c41/DRLND-Navigation/blob/master/Report.md).


#### Actor
Below, the flow diagram demonstrates how the Actor network is setup.

![image](https://github.com/Ohara124c41/DRL-Continuous_Control/blob/master/actor.PNG?raw=true)


#### Critic
Below, the flow diagram demonstrates how the Critic network is setup.

![image](https://github.com/Ohara124c41/DRL-Continuous_Control/blob/master/critic.PNG?raw=true)



### Code Implementation


**NOTE:** Code will run in GPU if CUDA is available, otherwise it will run in CPU

Code is structured in different modules. The most relevant features will be explained next:

1. **model.py:** It contains the main execution thread of the program. This file is where the main algorithm is coded (see *algorithm* above). PyTorch is utilized for training the agent in the environment. The agent has an Actor and Critic network.
2. **ddpg_agent.py:** The model script contains  the **DDPG agent**, a **Replay Buffer memory**, and the **Q-Targets** feature. A `learn()` method uses batches to handle the value parameters and update the policy.
3. **Continuous_Control.ipynb:** The Navigation Jupyter Notebook provides an environment to run the *Tennis* game, import dependencies, train the DDPG, visualize via Unity, and plot the results. The hyperparameters can be adjusted within the Notebook.


#### PyTorch Specifics


Documentation: PyTorch Version
------------------------------

.. autofunction:: spinup.ddpg_pytorch

Saved Model Contents: PyTorch Version
-------------------------------------

The PyTorch saved model can be loaded with ``ac = torch.load('path/to/model.pt')``, yielding an actor-critic object (``ac``) that has the properties described in the docstring for ``ddpg_pytorch``.

You can get actions from this model with

.. code-block:: python

    actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))



### DDPG Hyperparameters

The DQN agent uses the following parameters values (defined in ddpg_agent.py)

```
BUFFER_SIZE = int(1e6)  # Replay buffer size
BATCH_SIZE = 256        # Batch size #128
GAMMA = 0.99            # Discount Factor #0.99
TAU = 1e-3              # Soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
ACTOR_FC1_UNITS = 256   # Number of units for L1 in the actor model
ACTOR_FC2_UNITS = 128   # Number of units for L2 in the actor model
CRITIC_FCS1_UNITS = 256 # Number of units for L1 in the critic model
CRITIC_FC2_UNITS = 128  # Number of units for L2 in the critic model
BN_MODE = 0             # Use Batch Norm; 0=disabled, 1=BN before Activation, 2=BN after Activation (3, 4 are alt. versions of 1, 2)
ADD_OU_NOISE = True     # Toggle Ornstein-Uhlenbeck noisy relaxation process
THETA = 0.15            # k/gamma -> spring constant/friction coefficient [Ornstein-Uhlenbeck]
MU = 0.                 # x_0 -> spring length at rest [Ornstein-Uhlenbeck]
SIGMA = 0.2             # root(2k_B*T/gamma) -> Stokes-Einstein for effective diffision [Ornstein-Uhlenbeck]
ENV_STATE_SIZE = states.shape[1]
```

### Results

With the afformentioned setup, the agent was able to successfully meet the functional specifications in 500 episodes with an average score of 33.31 (see below):
```
Start training:
Episode 100	Average Score: 2.49	Score: 4.56
Episode 200	Average Score: 10.64	Score: 21.87
Episode 300	Average Score: 22.76	Score: 18.32
Episode 400	Average Score: 28.71	Score: 20.66
Environment solved in 500 episodes with an Average Score of 33.31
```


<img src="results.png" width="650">


### Future Work

This section contains two additional algorithms that would vastly improve over the current implementation, namely TRPO and TD3. Such algorithms have been developed to improve over DQNs and DDPGs.

- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477):
> We describe an iterative procedure for optimizing policies, with guaranteed monotonic improvement. By making several approximations to the theoretically-justified procedure, we develop a practical algorithm, called Trust Region Policy Optimization (TRPO). This algorithm is similar to natural policy gradient methods and is effective for optimizing large nonlinear policies such as neural networks. Our experiments demonstrate its robust performance on a wide variety of tasks: learning simulated robotic swimming, hopping, and walking gaits; and playing Atari games using images of the screen as input. Despite its approximations that deviate from the theory, TRPO tends to give monotonic improvement, with little tuning of hyperparameters.



- [Twin-Delay DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)

> Twin Delayed Deep Deterministic policy gradient algorithm (TD3), an actor-critic algorithm which considers the interplay between function approximation error in both policy and value updates. We evaluate our algorithm on seven continuous control domains from OpenAI gym (Brockman et al., 2016), where we outperform the state of the art by a wide margin. TD3 greatly improves both the learning speed and performance of DDPG in a number of challenging tasks in the continuous control setting.  Our algorithm exceeds the performance of numerous state of the art algorithms. As our modifications are simple to implement, they can be easily added to any other actor-critic algorithm.





## Additional References
_[1] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)_

_[2] [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1509.02971.pdf)._
