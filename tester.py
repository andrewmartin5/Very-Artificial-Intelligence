import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch

class Network(nn.Module):
    
    def __init__(self, n_observations, n_actions, *args, **kwargs) -> None:
        super().__init__()

        hidden1Size = 6
        hidden2Size = 6
                
        self.model = nn.Sequential(
            nn.Linear(n_observations, hidden1Size),
            nn.ReLU6(),
            nn.Linear(hidden1Size, hidden2Size),
            nn.ReLU6(),
            nn.Linear(hidden2Size, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

from algorithms.Gradient import Gradient

# Create and wrap the environment
env = gym.make('CartPole-v1', render_mode="human")

num_episodes = 1000
steps_per_ep = 1000
render = True

n_observations = env.observation_space.shape[0]
n_actions =  env.action_space.n
network = Network(n_observations, n_actions)
agent = Gradient(network)

for episode in range(num_episodes):
    
    obs, info = env.reset()
    end = False
    rewards = []
    actions = []
    states = []
    probs = []
    
    for step in range(steps_per_ep):
        if render:
            env.render()
            
        action_probs = network(torch.tensor(obs, dtype=torch.float32))
        action = torch.multinomial(action_probs, 1).item()
        obs, r, end, truncated, info = env.step(1)
        if end:
            break
        
        rewards.append(r)
        actions.append(action)
        states.append(obs)
        probs(action_probs)

    loss = agent.train(rewards, actions)


obs, info = env.reset()
