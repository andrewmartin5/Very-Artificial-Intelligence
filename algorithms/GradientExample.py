# Policy Gradient Method as implemented in https://github.com/microsoft/AI-For-Beginners/tree/main
import numpy as np
import torch
import torch.optim as optim

eps = 0.0001
alpha = 1e-4
lr = 0.01

def discounted_rewards(rewards,gamma=0.99,normalize=True):
    ret = []
    s = 0
    for r in rewards[::-1]:
        s = r + gamma * s
        ret.insert(0, s)
    if normalize:
        ret = (ret-np.mean(ret))/(np.std(ret)+eps)
    return ret


class Gradient:
    def __init__(self, network) -> None:
        self.network = network
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr)
        
    def train_on_batch(self, x, y):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        self.optimizer.zero_grad()
        predictions = self.network(x)
        loss = -torch.mean(torch.log(predictions) * y)
        loss.backward()
        self.optimizer.step()
        return loss
        
    def train(self, states, probs, actions, rewards):
        one_hot_actions = np.eye(2)[actions.T][0]
        gradients = one_hot_actions-probs
        dr = discounted_rewards(rewards)
        gradients *= dr
        target = alpha*np.vstack([gradients])+probs
        self.train_on_batch(states,target)