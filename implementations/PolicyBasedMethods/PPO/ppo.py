""" Trying to train PPO. References used for implementation:
Phil's video: https://youtu.be/hlv79rcHws0
video by wandb: https://youtu.be/MEt6rrxH8W4
Hyperparameters in Deep Reinforcement Learning that Matters: https://arxiv.org/pdf/1709.06560.pdf
Hyperparameters in IMPLEMENTATION MATTERS IN DEEP POLICY GRADIENTS: A CASE STUDY ON PPO AND TRPO: https://openreview.net/attachment?id=r1etN1rtPB&name=original_pdf
https://ppo-details.cleanrl.dev//2021/11/05/ppo-implementation-details/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

from models.models import Actor, Critic
from utils.memory import Memory
from utils.utils import calculate_advantage

import gym
from gym.vector import SyncVectorEnv
from gym.wrappers import RecordEpisodeStatistics
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import hydra
import logging
from tqdm import tqdm
import wandb
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)
c_handler = logging.StreamHandler()
c_format = logging.Formatter('%(name)s : %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
logger.propagate = False


wandb.init(project="ppo-Enhanced-CartPole-v1", entity="harshraj22", mode="disabled")

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    random.seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)
    torch.manual_seed(cfg.exp.seed)
    torch.backends.cudnn.deterministic = cfg.exp.torch_deterministic

    # so that the environment automatically resets
    env = SyncVectorEnv([
        lambda: RecordEpisodeStatistics(gym.make('CartPole-v1'))
    ])

    actor, critic = Actor(), Critic()
    actor_optim = Adam(actor.parameters(), eps=1e-5, lr=cfg.params.actor_lr)
    critic_optim = Adam(critic.parameters(), eps=1e-5, lr=cfg.params.critic_lr)
    memory = Memory(mini_batch_size=cfg.params.mini_batch_size, batch_size=cfg.params.batch_size)
    obs = env.reset()
    global_rewards = []

    NUM_UPDATES = (cfg.params.total_timesteps // cfg.params.batch_size) * cfg.params.epochs
    cur_timestep = 0

    def calc_factor(cur_timestep: int) -> float:
        """Calculates the factor to be multiplied with the learning rate to update it."""
        update_number = cur_timestep // cfg.params.batch_size
        total_updates = cfg.params.total_timesteps // cfg.params.batch_size
        fraction = 1.0 - update_number / total_updates
        return fraction

    actor_scheduler = LambdaLR(actor_optim, lr_lambda=calc_factor, verbose=True)
    critic_scheduler = LambdaLR(critic_optim, lr_lambda=calc_factor, verbose=True)

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
        obs_, reward, done, info = env.step(action)
        
        if done[0]:
            tqdm.write(f'Reward: {info[0]["episode"]["r"]}, Avg Reward: {np.mean(global_rewards[-10:]):.3f}')
            global_rewards.append(info[0]['episode']['r'])
            wandb.log({'Avg_Reward': np.mean(global_rewards[-10:]), 'Reward': info[0]['episode']['r']})

        memory.remember(obs.squeeze(0).cpu().numpy(), action.item(), log_prob.item(), reward.item(), done.item(), value.item())
        obs = obs_
        cur_timestep += 1

        # if the current timestep is a multiple of the batch size, then we need to update the model
        if cur_timestep % cfg.params.batch_size == 0:
            for epoch in tqdm(range(cfg.params.epochs), desc=f'Num updates: {cfg.params.epochs * (cur_timestep // cfg.params.batch_size)} / {NUM_UPDATES}'):
                # sample a batch from memory of experiences
                old_states, old_actions, old_log_probs, old_rewards, old_dones, old_values, batch_indices = memory.sample()
                old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
                old_actions = torch.tensor(old_actions, dtype=torch.float32)
                advantage = calculate_advantage(old_rewards, old_values, old_dones, gae_gamma=cfg.params.gae_gamma, gae_lambda=cfg.params.gae_lambda)
                
                advantage = torch.tensor(advantage, dtype=torch.float32)
                old_rewards = torch.tensor(old_rewards, dtype=torch.float32)
                old_values = torch.tensor(old_values, dtype=torch.float32)

                # for each mini batch from batch, calculate advantage using GAE
                for mini_batch_index in batch_indices:
                    # remember: Normalization of advantage is done on mini batch, not the entire batch
                    advantage[mini_batch_index] = (advantage[mini_batch_index] - advantage[mini_batch_index].mean()) / (advantage[mini_batch_index].std() + 1e-8)

                    dist = actor(torch.tensor(old_states[mini_batch_index], dtype=torch.float32).unsqueeze(0))
                    # actions = dist.sample()
                    log_probs = dist.log_prob(old_actions[mini_batch_index]).squeeze(0)
                    entropy = dist.entropy().squeeze(0)

                    log_ratio = log_probs - old_log_probs[mini_batch_index]
                    ratio = torch.exp(log_ratio)

                    with torch.no_grad():
                        # approx_kl = ((ratio-1)-log_ratio).mean()
                        approx_kl = ((old_log_probs[mini_batch_index] - log_probs)**2).mean()
                        wandb.log({'Approx_KL': approx_kl})

                    actor_loss = -torch.min(
                        ratio * advantage[mini_batch_index],
                        torch.clamp(ratio, 1 - cfg.params.actor_loss_clip, 1 + cfg.params.actor_loss_clip) * advantage[mini_batch_index]
                    ).mean()

                    values = critic(torch.tensor(old_states[mini_batch_index], dtype=torch.float32).unsqueeze(0)).squeeze(-1)
                    returns = old_values[mini_batch_index] + advantage[mini_batch_index]

                    critic_loss = torch.max(
                        (values - returns)**2,
                        (old_values[mini_batch_index] + torch.clamp(
                            values - old_values[mini_batch_index], -cfg.params.critic_loss_clip, cfg.params.critic_loss_clip
                            ) - returns
                        )**2
                    ).mean()
                    # critic_loss = F.mse_loss(values, returns)

                    wandb.log({'Actor_Loss': actor_loss.item(), 'Critic_Loss': critic_loss.item(), 'Entropy': entropy.mean().item()})
                    loss = actor_loss + 0.25 * critic_loss - 0.01 * entropy.mean()
                    actor_optim.zero_grad()
                    critic_optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), cfg.params.max_grad_norm)
                    nn.utils.clip_grad_norm_(critic.parameters(), cfg.params.max_grad_norm)

                    actor_optim.step()
                    critic_optim.step()

            memory.reset()
            actor_scheduler.step(cur_timestep)
            critic_scheduler.step(cur_timestep)

            y_pred, y_true = old_values.cpu().numpy(), (old_values + advantage).cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            wandb.log({'Explained_Var': explained_var})

    if cfg.exp.save_weights:
        torch.save(actor.state_dict(), Path(f'{hydra.utils.get_original_cwd()}/{cfg.exp.model_dir}/actor.pth'))
        torch.save(critic.state_dict(), Path(f'{hydra.utils.get_original_cwd()}/{cfg.exp.model_dir}/critic.pth'))


if __name__ == '__main__':
    main()