# Policy Gradient Method as implemented in https://github.com/microsoft/AI-For-Beginners/tree/main
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import math

lr = 1

def discounted_rewards(rewards,gamma=0.99,normalize=True):
    ret = []
    s = 0
    for r in rewards[::-1]:
        s = r + gamma * s
        ret.insert(0, s)
    if normalize:
        ret = (ret-np.mean(ret))/max(np.std(ret),0.0001)
    return ret


class Gradient:
    def __init__(self, network) -> None:
        self.network = network
        self.op = optim.Adam(self.network.parameters(), lr=0.1)

    def train_on_batch(self, input, target):
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        predictions = self.network(input)
        loss = -torch.mean(torch.log(predictions) * target)
        loss.backward()
        with torch.no_grad():
            for param in self.network.parameters():
                if param == None:
                    continue
                param -= lr * param.grad
        # self.op.step()
        self.network[0].weight.grad.zero_()
        self.network[0].bias.grad.zero_()
        self.network[2].weight.grad.zero_()
        self.network[2].bias.grad.zero_()
        return loss
        
    def train(self, states, probs, actions, rewards):
        """
        One-hot our actions (categorize into [1 0] for left and [0 1] for right) 
        We do this by taking the nth column of our identity matrix
        """
        one_hot_actions = np.eye(probs.shape[1])[actions.T][0]
        
        """
        Calculate gradient based on how "confident" we were on each action
        """
        gradients = one_hot_actions-probs
        
        """
        Discount rewards, weighting values with higher reward over those with lower reward
        """
        dr = discounted_rewards(rewards)
        
        """Discount each gradient to prioritize higher rewards"""
        gradients *= dr
        target = gradients# + probs
        self.train_on_batch(states,target)