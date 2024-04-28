import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
from algorithms.Gradient import Gradient
import os
 
def run_episode(env, episode_length, render=False):
    # Initialize lists of outputs
    states, actions, probs, rewards = [],[],[],[]
    state, _ = env.reset()
    
    for _ in range(episode_length):
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
episode_length = 5000

model = nn.Sequential(
    nn.Linear(num_inputs, 16),
    nn.ReLU6(),
    nn.Linear(16, num_actions),
    nn.Softmax(dim=-1)
)

agent = Gradient(model)

maxReward = 0

for _ in range(50000):
    states, probs, actions, rewards = run_episode(env, episode_length)
    agent.train(states, probs, actions, rewards)
    total = sum(rewards)
    if total == episode_length:
        break
    if total > maxReward:
        print(f"Best: {sum(rewards)}", end="\r")
        
os.system('spd-say "Model trained successfully"')


# Visually confirm output
input("Are you ready to see results?")

torch.save(model, 'adam.pt')

env = gym.make("CartPole-v1", render_mode="human")
_, _, _, rewards = run_episode(env, episode_length, render=True)
print(sum(rewards))
