# Policy Gradient Method
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class Gradient:
    def __init__(self, network) -> None:
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.99)
        
    def train(self, actions, rewards, probs, states):
        one_hot_actions = np.eye(2)[actions.T][0]
        gradients = one_hot_actions-probs
        dr = discounted_rewards(rewards)
        gradients *= dr
        target = alpha*np.vstack([gradients])+probs
        train_on_batch(states,target)
        history.append(np.sum(rewards))
        if epoch%100==0:
            print(f"{epoch} -> {np.sum(rewards)}")

        # rewards = torch.tensor(rewards)
        # self.optimizer.zero_grad()
        # loss =  -torch.sum(rewards)
        # # loss.backward()
        # self.optimizer.step()
        # return loss.item()