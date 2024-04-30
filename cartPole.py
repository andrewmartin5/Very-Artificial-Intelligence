import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
from algorithms.Gradient import Gradient
import os
import sys
from tqdm import tqdm, trange
import wandb
from wandb.keras import WandbCallback


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
        "learning_rate": 0.1,
        }
    )


    env = gym.make("CartPole-v1")
    renderEnv = gym.make("CartPole-v1", render_mode="human")
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = nn.Sequential(
        nn.Linear(num_inputs, 8),
        nn.ReLU6(),
        nn.Linear(8, 4),
        nn.ReLU6(),
        nn.Linear(4, num_actions),
        nn.Softmax(dim=-1)
    )

    agent = Gradient(model, learning_rate=wandb.config["learning_rate"])

    maxReward = 0
    counter = 0
    with tqdm(total=10000) as bar:
        for i in range(10000):
        # while True:
            if i % 1000 == 0:
                states, probs, actions, rewards = run_episode(env, model, episode_length)
            else:
                states, probs, actions, rewards = run_episode(env, model, episode_length)
            loss = agent.train(states, probs, actions, rewards)
            total = sum(rewards)[0]
            wandb.log({"acc": total, "loss": loss})

            bar.update(1)
            # if total > maxReward:
            #     bar.update(total - maxReward)
            #     maxReward = total
            # else:
            #     bar.refresh()
            # if total == episode_length:
            #     counter += 1
            # if counter > episode_length * 0.1:
            #     break

    # os.system('spd-say "Model trained successfully"')
    torch.save(model, 'MAE2.pt')
    # input("Are you ready to see results?")

else:
    model = torch.load("MAE2.pt")


    # Visually confirm output


    env = gym.make("CartPole-v1", render_mode="human")
    _, _, _, rewards = run_episode(env, model, 1000000, render=True)
    print(sum(rewards))
