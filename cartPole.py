import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
from algorithms.Gradient import Gradient
import os
import sys
from tqdm import trange
import wandb

episode_length = 1000

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

    return torch.tensor(np.vstack(states)), torch.tensor(np.vstack(probs)), np.vstack(actions), np.vstack(rewards)

# if len(sys.argv) <= 1:
if True:
    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="CartpoleCompare",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.8
        }
    )


    env = gym.make("CartPole-v1")
    renderEnv = gym.make("CartPole-v1", render_mode="human")
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = nn.Sequential(
        nn.Linear(num_inputs, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, num_actions),
        nn.Softmax(dim=-1)
    )

    agent = Gradient(model, learning_rate=wandb.config["learning_rate"], with_adam=False)

    maxReward = 0
    counter = 0

    rs = []

    for i in trange(1500):
    # while True:
        states, probs, actions, rewards = run_episode(env, model, episode_length)
        loss = agent.train(states, probs, actions, rewards)
        total = sum(rewards)[0]
        if len(rs) > 25:
            rs.pop(0)
        rs.append(total)
        wandb.log({"acc": sum(rs)/len(rs)})
        # wandb.log({"acc": total})

    torch.save(model, 'Grad.pt')

else:
    # Visually confirm output

    model = torch.load("Grad.pt")
    env = gym.make("CartPole-v1", render_mode="human")
    _, _, _, rewards = run_episode(env, model, 1000000, render=True)
    print(sum(rewards))
