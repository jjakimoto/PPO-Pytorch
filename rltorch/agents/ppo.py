from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

from .core import ACAgent
from ..layers import Flatten, ACModel


class PPOAgent(ACAgent):
    """Proximal Policy Optimization

    n_epochs: int
        The number of steps to fit models after collecting data
    clip_eps: float
        Clip parameter for loss function
    """
    def __init__(self, state_shape, action_config, processor=None,
                 reward_reshape=None, smooth_length=100, log_dir='./logs',
                 window_length=4, lr=7e-4, model_config=None,
                 action_dist=dist.Categorical, discount=0.99, gae_lambda=0.95,
                 num_frames_per_proc=32, batch_size=32, entropy_coef=0.01,
                 value_loss_coef=0.2,
                 max_grad_norm=None, clip_eps=0.2, n_epochs=4, **kwargs):

        super(PPOAgent, self).__init__(state_shape, action_config, processor,
                                       reward_reshape, smooth_length, log_dir,
                                       window_length, lr, model_config,
                                       action_dist, discount,
                                       gae_lambda, num_frames_per_proc,
                                       batch_size,
                                       entropy_coef, value_loss_coef,
                                       max_grad_norm, **kwargs)
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs

    def fit(self, *args, **kwargs):
        if self.memory.nb_states < self.num_frames_per_proc:
            return
        if self.debug:
            old_model = deepcopy(self.ac_model)
        # Switch devices for optimization
        self.ac_model.to(self.device)
        experience = self.aggregate_experiences()
        T = experience.advantage.size(0)
        for epoch in range(self.n_epochs):
            for idx in range(T // self.batch_size):
                t_st = idx * self.batch_size
                t_end = (idx + 1) * self.batch_size
                fix_advs = experience.advantage[t_st:t_end]
                fix_log_probs = experience.log_prob[t_st:t_end]
                batch_states = experience.state[t_st:t_end]
                batch_actions = experience.action[t_st:t_end]
                batch_targets = experience.target[t_st:t_end]
                # Actor Training
                dists, values = self.ac_model(batch_states)
                log_probs = dists.log_prob(batch_actions)
                ratio = torch.exp(log_probs - fix_log_probs)
                # Use advantage as constanats when optimizing the policy
                surr1 = ratio * fix_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps,
                                    1.0 + self.clip_eps) * fix_advs
                actor_loss = -torch.min(surr1, surr2).mean()
                # Critic Training
                critic_loss = ((batch_targets - values) ** 2).mean()
                # Entropy regularization
                entropy = dists.entropy().mean()
                # Total Loss
                loss = actor_loss \
                        + self.value_loss_coef * critic_loss \
                        - self.entropy_coef * entropy
                # Optimizer Model
                self.optimizer.zero_grad()
                # Need to keep intermediate results for multiple loops
                loss.backward()
                # Clip Gradient
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm(self.parameters, self.max_grad_norm)
                self.optimizer.step()
                self.loss_record.append(loss.item())
                self.actor_loss_record.append(actor_loss.item())
                self.critic_loss_record.append(critic_loss.item())
                self.entropy_record.append(entropy.item())
        self.writer.add_scalar(f'data/loss', np.mean(self.loss_record),
                               self.record_step)
        self.writer.add_scalar(f'data/actor_loss', np.mean(self.actor_loss_record),
                               self.record_step)
        self.writer.add_scalar(f'data/critic_loss', np.mean(self.critic_loss_record),
                               self.record_step)
        self.writer.add_scalar(f'data/entropy', np.mean(self.entropy_record),
                               self.record_step)
        for key in self.reward_record.keys():
            self.writer.add_scalar(f'data/reward_{key}',
                                   np.mean(self.reward_record[key]),
                                   self.record_step)
        self.ac_model.to(self.experience_device)
        if self.debug:
            self.is_updated_model_with_names(self.ac_model, old_model)

    def build_model(self, config=None):
        # Share layer
        model = nn.Sequential()
        in_features = self.state_shape[0]
        model.add_module('conv1', nn.Conv2d(in_features, 32, 8, stride=4))
        model.add_module('relu1', nn.ReLU())
        model.add_module('conv2', nn.Conv2d(32, 64, 4, stride=2))
        model.add_module('relu2', nn.ReLU())
        model.add_module('conv3', nn.Conv2d(64, 64, 3, stride=1))
        model.add_module('relu3', nn.ReLU())
        model.add_module('flatten', Flatten())
        # Calculate dimension after passing test data
        dim = self._calc_dim(model)
        # dim = 32 * 7 * 7
        # Fully connected layers
        model.add_module('fc1', nn.Linear(dim, 512))
        model.add_module('relu4', nn.ReLU())

        # Actor layer
        actor_model = nn.Sequential()
        actor_model.add_module('actor_fc', nn.Linear(512, self.action_config['n_action']))
        actor_model.add_module('actor_softmax', nn.Softmax())

        # Value layer
        value_model = nn.Sequential()
        value_model.add_module('value_fc', nn.Linear(512, 1))
        # Combine all models
        ac_model = ACModel(model, actor_model, value_model, self.action_dist)
        return ac_model
