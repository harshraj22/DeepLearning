## A PyTorch implementation of PPO on CartPole-v1

See latest experiments on [wandb](https://wandb.ai/harshraj22/ppo-Enhanced-CartPole-v1)

Run using `python3 ppo.py` after installing the dependencies.

![image](https://user-images.githubusercontent.com/46635452/148691623-9cc7828f-af49-4ad5-897b-d87d74eca1e9.png)





:v:😎 So, finally my implementation of PPO from scratch [solves](https://wandb.ai/harshraj22/ppo-Enhanced-CartPole-v1/runs/2yjamauv?workspace=user-harshraj22) CartPole-V1.
Here's my learnings from trying to debug the same code for more than a month, and reading lots of discussions on Reddit, Stackoverflow and other channels on slack.
- PPO on CartPole-V1 is a very simple task. It might be solved even without bells and whistles like GAE, LR scheduling etc. If your model is not able to constantly score even 150, there is definitely some bug in your implementation. Hyperparameters aren't bad, your implementation is.
- [PPO Series](https://www.youtube.com/playlist?list=PLD80i8An1OEHhcxclwq8jOMam0m0M9dQ_) by [Costa](https://github.com/vwxyzjn) is actually an excellent resource to understand. 
- Below are the bugs that I had made during my implementation:
```python
# for each mini batch from batch, calculate advantage using GAE
for mini_batch_index in batch_indices:
    advantage[mini_batch_index] = (advantage[mini_batch_index] - advantage[mini_batch_index].mean()) / (advantage[mini_batch_index].std() + 1e-8)

    dist = actor(torch.tensor(old_states[mini_batch_index], dtype=torch.float32).unsqueeze(0))
    # actions = dist.sample() 
    # log_probs = dist.log_prob(actions).squeeze(0)<-- Remember, when calculating the log_prob to update the actor, the log_prob is to
    # be calculated with respect to old actions that were stored during playing game. You don't have to generate new
    # actions. This was my first Bug
    log_probs = dist.log_prob(old_actions[mini_batch_index]).squeeze(0)
    entropy = dist.entropy().squeeze(0)
    # calculate actor & critic loss, and update the model
```

```python
obs = env.reset()
with torch.no_grad():
    dist = actor(obs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    value = critic(obs)
obs_, reward, done, info = env.step(action)
# Note that we are storing 'obs' and not 'obs_'. the old observation is stored
# in the momory, not the newly generated one after taking the 'action'
memory.remember(obs, action, log_prob, reward, done, value)
obs = obs_

```
<br>

ToDo:
- [ ] Improve the code to make run the code with more that one parallel environments
- [ ] make it trainable on GPU (ie. add `.to(device)` appropriately)
- [ ] Write scripts for just inference, loading the saved weights. Use Hydra for Command Line Arg
