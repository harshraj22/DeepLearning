""" Trying to train PPO. References used for implementation:
Phil's video: https://youtu.be/hlv79rcHws0
video by wandb: https://youtu.be/MEt6rrxH8W4
Hyperparameters in Deep Reinforcement Learning that Matters: https://arxiv.org/pdf/1709.06560.pdf
Hyperparameters in IMPLEMENTATION MATTERS IN DEEP POLICY GRADIENTS: A CASE STUDY ON PPO AND TRPO: https://openreview.net/attachment?id=r1etN1rtPB&name=original_pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

from models.models import Actor, Critic
from utils.memory import Memory
from utils.utils import calculate_advantage

import gym
from gym.vector import SyncVectorEnv
from gym.wrappers import RecordEpisodeStatistics
import hydra
import logging
from tqdm import tqdm
import wandb
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)
c_handler = logging.StreamHandler()
c_format = logging.Formatter('%(name)s : %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

wandb.init(project="ppo-Enhanced-CartPole-v1", entity="harshraj22")#, mode="disabled")

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    random.seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)
    torch.manual_seed(cfg.exp.seed)
    torch.backends.cudnn.deterministic = cfg.exp.torch_deterministic

    # so that the environment automatically resets
    env = SyncVectorEnv([lambda: RecordEpisodeStatistics(gym.make('CartPole-v1'))])
    actor, critic = Actor(), Critic()
    actor_optim = Adam(actor.parameters(), eps=1e-5, lr=cfg.params.actor_lr)
    critic_optim = Adam(critic.parameters(), eps=1e-5, lr=cfg.params.critic_lr)
    memory = Memory(mini_batch_size=cfg.params.mini_batch_size, batch_size=cfg.params.batch_size)
    obs = env.reset()
    global_rewards = []

    NUM_UPDATES = cfg.params.total_timesteps // cfg.params.batch_size
    cur_timestep = 0

    while cur_timestep < cfg.params.total_timesteps:
        # keep playing the game
        obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist = actor(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = critic(obs)
        action = action.cpu().numpy()
        value = value.cpu().numpy()
        log_prob = log_prob.cpu().numpy()
        obs, reward, done, info = env.step(action)
        # logger.info(f'obs: {obs}, action: {action}, log_prob: {log_prob} reward: {reward}, done: {done}, value: {value}')
        # logger.info(f'obs: {type(obs)}, action: {type(action)}, log_prob: {type(log_prob)} reward: {type(reward)}, done: {type(done)}, value: {type(value)}')
        # logger.info(f'obs: {obs.shape}, action: {action.shape}, log_prob: {log_prob.shape} reward: {reward.shape}, done: {done.shape}, value: {value.shape}')
        # exit(0)
        if done[0]:
            tqdm.write(f'{info}')
            global_rewards.append(info[0]['episode']['r'])
            wandb.log({'Avg_Reward': np.mean(global_rewards[-10:]), 'Reward': info[0]['episode']['r']})
        memory.remember(obs[0], action.item(), log_prob.item(), reward.item(), done.item(), value.item())
        cur_timestep += 1

        # if len(memory.rewards) > 100:
        #     tqdm.write(f'Average Score: {np.mean(memory.rewards[-100:])} | Memory Size: {len(memory.rewards)}')

        # if the current timestep is a multiple of the batch size, then we need to update the model
        if cur_timestep % cfg.params.batch_size == 0:
            for epoch in tqdm(range(cfg.params.epochs), desc=f'Current timestep: {cur_timestep} / {cfg.params.total_timesteps}'):
                # sample a batch from memory of experiences
                old_states, old_actions, old_log_probs, old_rewards, old_dones, old_values, batch_indices = memory.sample()
                # np.random.shuffle(batch_indices)
                old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
                advantage = calculate_advantage(old_rewards, old_values, old_dones, gae_gamma=cfg.params.gae_gamma, gae_lambda=cfg.params.gae_lambda)
                # normalize the advantage
                # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
                advantage = torch.tensor(advantage, dtype=torch.float32)
                old_rewards = torch.tensor(old_rewards, dtype=torch.float32)
                old_values = torch.tensor(old_values, dtype=torch.float32)
                
                # logger.info(f'old_states: {old_states.shape}, old_actions: {old_actions.shape}, old_log_probs: {old_log_probs.shape}, old_rewards: {old_rewards.shape}, old_dones: {old_dones.shape}, old_values: {old_values.shape}')

                # for each mini batch from batch, calculate advantage using GAE
                for mini_batch_index in batch_indices:

                    # update actor and critic
                    dist = actor(torch.tensor(old_states[mini_batch_index], dtype=torch.float32).unsqueeze(0))
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions).squeeze(0)
                    entropy = dist.entropy().squeeze(0)

                    log_ratio = log_probs - old_log_probs[mini_batch_index]
                    ratio = torch.exp(log_ratio)

                    with torch.no_grad():
                        approx_kl = ((ratio-1)-log_ratio).mean()
                        wandb.log({'Approx_KL': approx_kl})

                    actor_loss = -torch.min(
                        ratio * advantage[mini_batch_index],
                        torch.clamp(ratio, 1 - cfg.params.actor_loss_clip, 1 + cfg.params.actor_loss_clip) * advantage[mini_batch_index]
                    ).mean()

                    critic_loss = F.mse_loss(
                        critic(torch.tensor(old_states[mini_batch_index], dtype=torch.float32).squeeze(0)).squeeze(-1),
                        old_values[mini_batch_index] + advantage[mini_batch_index]
                    )

                    # print(actor_loss, critic_loss)
                    # tqdm.write(f'{ratio}')
                    wandb.log({'Actor_Loss': actor_loss.item(), 'Critic_Loss': critic_loss.item(), 'Entropy': entropy.mean().item()})
                    loss = actor_loss + 0.5 * critic_loss - 0.08 * entropy.mean()
                    actor_optim.zero_grad()
                    critic_optim.zero_grad()
                    loss.backward()
                    actor_optim.step()
                    critic_optim.step()
            memory.reset()

    # for update in range(NUM_UPDATES):
    #     # for N epochs
    #     for _ in range(cfg.params.epochs):



    # observation = env.reset()
    # observation, reward, done, info = env.step([0])
    # logger.info(f'observation: {observation.shape}, reward: {reward.shape}, done: {done.shape}, info: {info}')



if __name__ == '__main__':
    main()
