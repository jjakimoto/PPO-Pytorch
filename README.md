# PPO PyTorch
Implementation of PPO with PyTorch

# Installation
To use this repository, you need to install through `setup.py`.
```buildoutcfg
python setup.py install
```

After installation, you can use the file with `import rltorch`.

# Examples
```python
import gym

from rltorch import Runner
from rltorch.agents import PPOAgent
from rltorch.processors import AtariProcessor

env = gym.make('Breakout-v0').unwrapped

FRAME_WIDTH = 84
FRAME_HEIGHT = 84
WINDOW_LENGTH = 4
# state_shape = env.observation_space.shape
state_shape = (WINDOW_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)
action_config = {'n_action': env.action_space.n, 'type': 'integer'}
processor = AtariProcessor(FRAME_WIDTH, FRAME_HEIGHT)

# Define agent
agent = PPOAgent(state_shape, action_config, processor=processor,
                 window_length=WINDOW_LENGTH, n_epochs=5,
                 lr=2.5e-4, entropy_coef=0.01, value_loss_coef=1,
                 num_frames_per_proc=128)

# Define execution
runner = Runner(env, agent, num_workers=4, multi=True)

# Start running
optimzeid_agent = runner.simulate(training=True, notebook=True, render_freq=4)

```

# Refrences
### Implementation
* [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)
* [baselines](https://github.com/openai/baselines)

### Theory
* [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
* [HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION](https://arxiv.org/pdf/1506.02438.pdf)