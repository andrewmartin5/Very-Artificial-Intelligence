from generations import DarwinianShenanigans
import gymnasium as gym
import torch

POP_SIZE = 50
NUM_GENERATIONS = 30
NUM_SURVIVORS = 5
MUTATION_RATE = 0.1
LEN_EPISODE = 1000
LEN_FINAL_TEST = None


d = DarwinianShenanigans(POP_SIZE, gym.make("CartPole-v1"))
d.simulation(NUM_GENERATIONS, NUM_SURVIVORS, MUTATION_RATE)
best = max(d.population, key=lambda x: d.calcFitness(x, forcedCap=LEN_EPISODE, render=False))
best = torch.save(best, "Genetic.pt")

# best = torch.load("Grad.pt")
# best = torch.load("Adam.pt")
# best = torch.load("Genetic.pt")

scores = []
for i in range(10):
    env = gym.make("CartPole-v1")
    d.setEnv(env)
    # print(d.calcFitness(best, forcedCap=LEN_FINAL_TEST, render=False))
    scores.append(d.calcFitness(best, forcedCap=100000, render=False))
    env.close()

print(sum(scores) / len(scores))