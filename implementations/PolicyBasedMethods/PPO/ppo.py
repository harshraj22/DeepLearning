import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
import logging
import gym
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.CRITICAL)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        """Assuming the Actor for CartPole-v1 """
        
        self.l1 = nn.Linear(4, 2)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(2, 2)

    def forward(self, state):
        action = self.relu(self.l1(state))
        action = F.softmax(self.l2(action), dim=-1)
        return Categorical(action)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        """Assuming the Critic for CartPole-v1 """

        self.l1 = nn.Linear(4, 2)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(2, 1)

    def forward(self, state):
        value = self.relu(self.l1(state))
        return self.l2(value)


class Memory:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []

    def remember(self, state, action, log_prob, reward):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def merge(self, states, actions, log_probs, rewards):
        for state, action, log_prob, reward in zip(states, actions, log_probs, rewards):
            self.remember(state, action, log_prob, reward)
        
    def get_all(self):
        return self.states, self.actions, self.log_probs, self.rewards


# initialize all
actor = Actor()
critic = Critic()
actor_optim = Adam(actor.parameters())
critic_optim = Adam(critic.parameters())
env = gym.make('CartPole-v1')
memory = Memory()


def compute_r2g(rewards, gamma=0.9):
    """Given a rewards array, calculates the rewards-to-go (r2g), where
    r2g[i] = rewards[i] + sum(
        gamma**j * rewards[j] for j in range(i+1, len(rewards))
    )

    Args:
        rewards (List[Int]): The rewards collected in a trajectory
        gamma (float, optional): The discount factor. Defaults to 0.9.

    Returns:
        List[Int]: The Rewards-to-go as calculated by above formula.
    """
    r2g, cumm_reward = [], 0
    for reward in reversed(rewards):
        cumm_reward = reward + gamma*cumm_reward
        r2g.append(cumm_reward)

    return reversed(r2g)


def play_single_game(env, actor: Actor, memory: Memory):
    """Plays a single game in the environment and saves the trajectory in the
    memory.

    Args:
        env (gym.env): The gym environment
        actor (Actor): The actor, a neural net returning distributions
        memory (Memory): The memory to save the trajectories.
    """
    obs, done = env.reset(), False
    states, actions, log_probs, rewards = [], [], [], []
    # logging.debug(f'Shape of obs: {obs.shape}')

    while not done:
        obs = torch.as_tensor(obs, dtype=torch.float32)
        dist = actor(obs)
        # logging.debug(f'Shape of dist:, {dist}')
        action = dist.sample()
        # logging.debug(f'shape of action: {action.shape}: {action}')
        log_prob = dist.log_prob(action)
        obs, reward, done, info = env.step(action.item())
        states.append(obs) # .detach())
        rewards.append(reward) # .item())
        log_probs.append(log_prob.detach())
        actions.append(action)
    
    r2g = compute_r2g(rewards)
    memory.merge(states, actions, log_probs, r2g)

    return sum(rewards)


def update(memory: Memory, actor: Actor, actor_optim: Adam, critic: Critic, critic_optim: Adam, eps=0.2):
    # update the actor & critic parameters
    actor_optim.zero_grad()
    critic_optim.zero_grad()

    # calculate Advantage (r2g - Value)
    states, old_actions, old_log_probs, rewards = memory.get_all() 
    old_log_probs = torch.stack(old_log_probs)
    states = torch.tensor(np.array(states))
    values = critic(states).view(-1) 

    rewards = np.array(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
    rewards = torch.tensor(rewards, requires_grad=True, dtype=torch.float32)

    advantage = rewards - values

    dist = actor(states)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)

    probs_ratio = log_probs / old_log_probs

    actor_loss = -torch.min(
        probs_ratio * advantage,
        torch.clip(probs_ratio, 1-eps, 1+eps) * advantage
    ).mean()


    critic_loss = F.mse_loss(values, rewards)
    logging.debug(f'Shapes: \nstates: {states.shape}\nvalues: {values.shape}, {values.requires_grad}\nrewards: {rewards.shape}, {rewards.requires_grad}\nadvantage: {advantage.shape}\n probs_ratio: {probs_ratio.shape}')

    loss = actor_loss + critic_loss
    logging.debug(f'Loss: {loss}, actor_loss: {actor_loss}, critic_loss: {critic_loss}')
    loss.backward()
    actor_optim.step()
    critic_optim.step()


NUM_GAMES = 2000
all_rewards = []
for i in tqdm(range(NUM_GAMES)):
    # play game and save trajectories
    current_reward = play_single_game(env, actor, memory)
    all_rewards.append(current_reward)

    # every 5th game, update the actor and critic
    if i%5 == 0:
        update(memory, actor, actor_optim, critic, critic_optim)
        memory.clear()
        tqdm.write(f'Current Reward: {current_reward}')

env.close()
plt.plot(all_rewards)
plt.savefig('fig.jpg')
