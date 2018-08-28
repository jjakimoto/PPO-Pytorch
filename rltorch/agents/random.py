import numpy as np

from .core import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, state_shape, action_config):
        super(RandomAgent, self).__init__(state_shape, action_config)

    def predict(self, observation):
        if self.action_config['type'] == 'integer':
            shape = self.action_config.get('shape', None)
            n_action = self.action_config['n_action']
            actions = np.random.randint(0, n_action, shape)
        else:
            raise NotImplementedError
        return actions