import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
from algorithms.Gradient import Gradient
import os
import sys
from tqdm import tqdm

episode_length = 5000
 
def run_episode(env, model, episode_length, render=False):
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

if len(sys.argv) <= 1:

    env = gym.make("CartPole-v1")
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = nn.Sequential(
        nn.Linear(num_inputs, 16),
        nn.ReLU6(),
        nn.Linear(16, num_actions),
        nn.Softmax(dim=-1)
    )

    agent = Gradient(model)

    maxReward = 0
    counter = 0
    with tqdm(total=episode_length) as bar:
        for _ in range(1000000):
            states, probs, actions, rewards = run_episode(env, model, episode_length)
            agent.train(states, probs, actions, rewards)
            total = sum(rewards)[0]
            if total > maxReward:
                bar.update(total - maxReward)
                maxReward = total
            else:
                bar.refresh()
            if total == episode_length:
                counter += 1
            if counter > episode_length * 0.01:
                break
            
    os.system('spd-say "Model trained successfully"')
    torch.save(model, 'tensor.pt')
    input("Are you ready to see results?")

else:
    model = torch.load("tensor.pt")


# Visually confirm output


env = gym.make("CartPole-v1", render_mode="human")
_, _, _, rewards = run_episode(env, model, episode_length, render=True)
print(sum(rewards))
