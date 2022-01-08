import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt


def calculate_advantage(rewards: npt.ArrayLike, values: npt.ArrayLike, dones: npt.ArrayLike, gae_gamma=0.99, gae_lambda=0.95) -> npt.ArrayLike:
    advantage = np.zeros_like(rewards)

    # rewards.shape: (batch_size)
    for t, _ in enumerate(rewards):
        discount, a_t = 1, 0
        # calculate advantage at timestep t
        for k in range(t, len(rewards)-1):
            a_t += discount * (rewards[k] + gae_gamma * values[k+1] * (1-int(dones[k])) - values[k])
            discount *= gae_gamma * gae_lambda

        advantage[t] = a_t
    
    return advantage