import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Layer Initialization as suggested here: https://www.youtube.com/watch?v=MEt6rrxH8W4&list=PLD80i8An1OEHhcxclwq8jOMam0m0M9dQ_&index=1

    Args:
        layer (nn.Linear): an instance of linear layer
        std (float, optional): Standard deviation for layer initialization. Defaults to np.sqrt(2).
        bias_const (float, optional): bais value for layer initialization. Defaults to 0.0.

    Returns:
        nn.Linear: Linear layer with weights and biases initialized
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# To Do: Check BatchNorm
class Actor(nn.Module):
    def __init__(self, in_dim=4, out_dim=2):
        super(Actor, self).__init__()
        """Assuming the Actor for CartPole-v1 """
        self.l1 = nn.Sequential(
                layer_init(nn.Linear(in_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, out_dim), std=0.01),
                nn.Softmax(dim=-1)
            )
        # self.l2 = layer_init(nn.Linear(64, 2), std=0.01)

    def forward(self, state):
        action = self.l1(state)
        # action = F.softmax(self.l2(action), dim=-1)
        return Categorical(action)


class ActorContinious(nn.Module):
    def __init__(self, in_dim=4, out_dim=2):
        super(ActorContinious, self).__init__()
        """Assuming the Actor for CartPole-v1 """
        self.l1 = nn.Sequential(
                layer_init(nn.Linear(in_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, out_dim), std=0.01),
            )

        self.log_std = nn.Parameter(torch.zeros(out_dim))
        # self.l2 = layer_init(nn.Linear(64, 2), std=0.01)

    def forward(self, state):
        means = self.l1(state)
        log_stds = self.log_std
        action_std = torch.exp(log_stds)
        dist = Normal(means, action_std)

        return dist


class Critic(nn.Module):
    def __init__(self, in_dim=4):
        super(Critic, self).__init__()
        """Assuming the critic for CartPole-v1"""
        self.l1 = nn.Sequential(
                layer_init(nn.Linear(in_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0)
            )
        # self.l2 = layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, state):
        value = self.l1(state)
        return value
        # return self.l2(value)