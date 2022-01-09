import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
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
    def __init__(self):
        super(Actor, self).__init__()
        """Assuming the Actor for CartPole-v1 """
        self.l1 = nn.Sequential(
                layer_init(nn.Linear(4, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 2), std=0.01),
                nn.Softmax(dim=-1)
            )
        # self.l2 = layer_init(nn.Linear(64, 2), std=0.01)

    def forward(self, state):
        action = self.l1(state)
        # action = F.softmax(self.l2(action), dim=-1)
        return Categorical(action)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        """Assuming the critic for CartPole-v1"""
        self.l1 = nn.Sequential(
                layer_init(nn.Linear(4, 64)),
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