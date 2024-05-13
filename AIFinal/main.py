from generations import DarwinianShenanigans
import gymnasium as gym
import torch

d = DarwinianShenanigans(0, gym.make("CartPole-v1"))

load = input("Which Model to Load? (grad/gen/adam)")

match load:
    case "grad":
        best = torch.load("Grad.pt")
    case "gen":
        best = torch.load("Genetic.pt")
    case "adam":
        best = torch.load("Adam.pt")
    case _:
        exit(0)

env = gym.make("CartPole-v1", render_mode="human")
d.setEnv(env)
print(d.calcFitness(best, forcedCap=None, render=False))
env.close()
