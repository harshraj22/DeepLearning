import numpy as np

class Memory:
    def __init__(self, mini_batch_size, batch_size):
        """Implements the memory class used to store the generated experiences
        while interacting with the environment and sampling data points from
        it.

        Args:
            mini_batch_size (Int): size of mini batch used for gradient descent
              while updating the Actor and Critic networks.
            batch_size (Int): Size of the batch used to sample experiences for 
              each epoch.
        """
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

        self.mini_batch_size = mini_batch_size
        self.batch_size = batch_size

    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def remember(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def merge(self, states, actions, log_probs, rewards, dones, values):
        for state, action, log_prob, reward, done, value in zip(states, actions, log_probs, rewards, dones, values):
            self.remember(state, action, log_prob, reward, done, value)
    
    def sample(self):
        indices = np.arange(0, self.batch_size)
        np.random.shuffle(indices)

        mini_batch_indices = [indices[start: start+self.mini_batch_size] for start in range(0, self.batch_size, self.mini_batch_size)]
        return np.array(self.states), np.array(self.actions), np.array(self.log_probs), np.array(self.rewards), np.array(self.dones), np.array(self.values), mini_batch_indices

