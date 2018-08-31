import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Identity(nn.Module):
    def forward(self, x):
        return x


class ACModel(nn.Module):
    def __init__(self, share_model, actor_model, value_model, action_dist=None):
        super(ACModel, self).__init__()
        self.share_model = share_model
        self.actor_model = actor_model
        self.value_model = value_model
        self.action_dist = action_dist

    def forward(self, x):
        x = self.share_model(x)
        action = self.actor_model(x)
        value = self.value_model(x)
        if self.action_dist is not None:
            action = self.action_dist(action)
        return action, value
