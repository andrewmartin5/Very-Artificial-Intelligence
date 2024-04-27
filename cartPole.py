import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
from algorithms.Gradient import Gradient

def run_episode(env, render=False):
    # Initialize lists of outputs
    states, actions, probs, rewards = [],[],[],[]
    state, _ = env.reset()
    
    for _ in range(1000):
        if render:
            env.render()
        
        # Get probability of each action from our model
        action_probs = model(torch.tensor(state, dtype=torch.float32)).detach()
        # Use multinomial to quickly choose weighted random action
        action = torch.multinomial(action_probs, 1).item()
        
        # Run step
        observation, reward, done, _, _ = env.step(action)
        if done:
            break
        
        # Update episode's history
        states.append(state)
        actions.append(action)
        probs.append(action_probs.numpy())
        rewards.append(reward)
        
        state = observation

    return np.vstack(states), np.vstack(probs), np.vstack(actions), np.vstack(rewards)

env = gym.make("CartPole-v1")
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n

model = nn.Sequential(
    nn.Linear(num_inputs, 64),
    nn.ReLU6(),
    nn.Linear(64, num_actions),
    nn.Softmax(dim=-1)
)

agent = Gradient(model)

for _ in range(200):
    states, probs, actions, rewards = run_episode(env)
    agent.train(states, probs, actions, rewards)
        
# Visually confirm output
env = gym.make("CartPole-v1", render_mode="human")
run_episode(env,render=True)
