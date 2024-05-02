from generations import DarwinianShenanigans
import gymnasium as gym

POP_SIZE = 50
NUM_GENERATIONS = 30
NUM_SURVIVORS = 5
MUTATION_RATE = 0.1
LEN_EPISODE = 1000
LEN_FINAL_TEST = None


d = DarwinianShenanigans(POP_SIZE, gym.make("CartPole-v1"))
d.simulation(NUM_GENERATIONS, NUM_SURVIVORS, MUTATION_RATE)
best = max(d.population, key=lambda x: d.calcFitness(x, forcedCap=LEN_EPISODE, render=False))

env = gym.make("CartPole-v1", render_mode='human')
d.setEnv(env)
print(d.calcFitness(best, forcedCap=LEN_FINAL_TEST, render=False))
env.close()