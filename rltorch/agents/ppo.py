import torch
import torch.nn as nn

from .core import ACAgent


class PPOAgent(ACAgent):
    """Proximal Policy Optimization

    n_epochs: int
        The number of steps to fit models after collecting data
    clip_eps: float
        Clip parameter for loss function
    """
    def __init__(self, state_shape, action_config, processor=None,
                 reward_reshape=None, smooth_length=100, log_dir='./logs',
                 window_length=4, lr=7e-4, critic_config=None,
                 actor_config=None,
                 action_dist=dist.Categorical, discount=0.99, gae_lambda=0.95,
                 num_frames_per_proc=32, batch_size=32, entropy_coef=0.01,
                 value_loss_coef=0.2,
                 max_grad_norm=None, clip_eps=0.2, n_epochs=4):

        super(PPOAgent, self).__init__(state_shape, action_config, processor,
                                       reward_reshape, smooth_length, log_dir,
                                       window_length, lr, critic_config,
                                       actor_config, action_dist, discount,
                                       gae_lambda, num_frames_per_proc,
                                       batch_size,
                                       entropy_coef, value_loss_coef,
                                       max_grad_norm)
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs

    def fit(self, *args, **kwargs):
        if self.memory.nb_states < self.num_frames_per_proc:
            return
        advs, log_probs, entropies = self.aggregate_experiences()
        n_actors = advs.size(1)
        pl_batch_size = self.batch_size * n_actors
        advs = advs.view(-1)
        log_probs = log_probs.view(-1)
        entropies = entropies.view(-1)
        idx = np.arange(advs.size(0))
        for epoch in range(self.n_epochs):
            # Actor Training
            sampled_idx = np.random.choice(idx, size=pl_batch_size,
                                           replace=False)
            batch_log_probs = log_probs[sampled_idx]
            batch_advs = advs[sampled_idx]
            batch_entropies = entropies[sampled_idx]
            old_log_probs = batch_log_probs.detach()
            ratio = torch.exp(batch_log_probs - old_log_probs)
            # Use advantage as constanats when optimizing the policy
            advs_const = batch_advs.detach()
            surr1 = ratio * advs_const
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps,
                                1.0 + self.clip_eps) * advs_const
            actor_loss = -torch.min(surr1, surr2).mean()
            # Critic Training
            batch_critic_loss = (batch_advs ** 2).mean()
            # Entropy regularization
            batch_entropy = batch_entropies.mean()
            # Total Loss
            loss = actor_loss \
                   + self.value_loss_coef * batch_critic_loss \
                   - self.entropy_coef * batch_entropy
            # Optimizer Model
            self.optimizer.zero_grad()
            if epoch == self.n_epochs - 1:
                loss.backward()
            else:
                # Need to keep intermediate results for multiple loops
                loss.backward(retain_graph=True)
            # Clip Gradient
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.parameters, self.max_grad_norm)
            self.optimizer.step()
            self.loss_record.append(loss.item())
        self.writer.add_scalar(f'data/loss', np.mean(self.loss_record),
                               self.record_step)
        for key in self.reward_record.keys():
            self.writer.add_scalar(f'data/reward_{key}',
                                   np.mean(self.reward_record[key]),
                                   self.record_step)

    def build_critic(self, config=None):
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
        # Fully connected layers
        model.add_module('fc1', nn.Linear(dim, 512))
        model.add_module('relu4', nn.ReLU())
        model.add_module('fc2', nn.Linear(512, 1))
        return model

    def build_actor(self, config=None):
        model = nn.Sequential()
        in_features = self.state_shape[0]
        model.add_module('conv1', nn.Conv2d(in_features, 32, 8, stride=4))
        model.add_module('relu1', nn.ReLU())
        model.add_module('conv2', nn.Conv2d(32, 64, 4, stride=2))
        model.add_module('relu2', nn.ReLU())
        model.add_module('conv3', nn.Conv2d(64, 64, 3, stride=1))
        model.add_module('relu3', nn.ReLU())
        model.add_module('flatten', Flatten())
        # Calculate dimention after passing test data
        dim = self._calc_dim(model)
        # Fully connected layers
        model.add_module('fc1', nn.Linear(dim, 512))
        model.add_module('relu4', nn.ReLU())
        model.add_module('fc2', nn.Linear(512, self.action_config['n_action']))
        model.add_module('softmax', nn.Softmax())
        return model
