import torch
import gymnasium as gym
from chromosome import Chromosome
import numpy as np
from tqdm import tqdm
import wandb


wandb.init(
    # set the wandb project where this run will be logged
    project="CartpoleCompare",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.1,
    }
)

class DarwinianShenanigans:

    def __init__(self, popSize, env: gym.Env) -> None:
        self.env = env
        self.inputSize = env.observation_space.shape[0]
        self.outputSize = env.action_space.n
        self.popSize = popSize
        self.population = [Chromosome(self.inputSize, self.outputSize) for _ in range(self.popSize)]
    
    def setEnv(self, env):
        self.env = env

    def calcFitness(self, c, forcedCap=1000, render=False):
        fitness = 0
        done = False
        with torch.no_grad(): # is there a way to get rid of this without fucking everything else up
            state, _ = self.env.reset() # initial state
            
            while not done:
                if render:
                    self.env.render()
                state = torch.tensor(state, dtype=torch.float32) # convert Observation object to tensor
                actionProbabilities = c(state).detach() # chuck the state into the chromosome
                # Andrew says multinomial is faster, so I'll take his word for it
                action = torch.multinomial(actionProbabilities, 1).item() 

                state, reward, done, _, _ = self.env.step(action)
                fitness += reward

                if forcedCap is not None and fitness == forcedCap: # breaks out of loop when cap is reached
                    done = True
        return fitness
                
    def simulation(self, numGenerations, numSurvivors, mutationRate):
        def sex(p1, p2):
            child = Chromosome(self.inputSize, self.outputSize)

            # this loops over all the weights and biases
            for childParams, p1Params, p2Params in zip(child.parameters(), p1.parameters(), p2.parameters()):
                # this is Ruml's suggested coinflip technique, here heads and tails act as a mask for p1 and p2's stuff
                heads = torch.randint(2, childParams.size()).float()
                tails = 1 - heads
                childParams.data = (p1Params.data * heads) + (p2Params * tails)

                # determine which weights and biases get mutated based on mutation rate
                temp = torch.rand_like(childParams) # generate probabilities
                temp[temp > mutationRate] = 0
                temp[temp != 0] = 1
                temp = temp * torch.randn_like(childParams) # randn is normally distributed, unlike rand
                childParams.data += temp
            return child

        with tqdm(total=numGenerations * self.popSize) as bar:
            for generation in range(numGenerations):
                fitnesses = [self.calcFitness(c) for c in self.population]
                for f in fitnesses:
                    wandb.log({"acc": max(fitnesses)})

                survivorsIndexes = np.argsort(fitnesses)[-numSurvivors:] # gets the n best of the current pop
                survivors = [self.population[i] for i in survivorsIndexes]
                newPop = []
                for _ in range(self.popSize):
                    parent1 = np.random.choice(survivors)
                    parent2 = np.random.choice(survivors)
                    newPop.append(sex(parent1, parent2))
                    bar.update(1)
                self.population = newPop



