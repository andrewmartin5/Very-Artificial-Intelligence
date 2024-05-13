# Vanilla policy gradient method, rebuilt but using similar process
import numpy as np
import torch
import torch.optim as optim

def discount(rewards,gamma=0.99,normalize=True):
    ret = np.zeros_like(rewards, dtype=float)
    s = 0
    for i, r in enumerate(rewards[::-1]):
        s = r + gamma * s
        ret[-(i+1)] = s
    if normalize:
        ret = (ret - np.mean(ret)) / max(np.std(ret), 0.0001)
    return ret


class Gradient:
    def __init__(self, network, learning_rate, with_adam = False) -> None:
        self.network = network
        self.alpha = learning_rate
        self.optimized = with_adam
        if with_adam:
            self.optimizer = optim.Adam(self.network.parameters(), lr = 0.01)
        
    def train(self, states, output, actions, rewards):
        """
        One-hot our actions (categorize into [1 0] for left and [0 1] for right) 
        We do this by taking the nth column of our identity matrix
        """
        one_hot_actions = torch.eye(output.shape[1])[actions.T][0]
        
        """
        Seed our advantage estimate given our model's confidence
        """
        advantage = one_hot_actions-output
        
        """
        Discount rewards, weighting values with higher reward over those with lower reward
        """
        dr = discount(rewards)
        
        """Discount each advantage to prioritize higher rewards"""
        advantage *= dr
        
        """Create a target that has our current probabilities plus our new advantage"""
        target = output + advantage
        
        """Recalculate probabilities of actions, used for backprop"""
        output = self.network(states)
        
        """Different Loss (error) functions will affect the way our network will respond to error"""
        # loss = -torch.mean(torch.log(output) * target) # Negative Log Loss function
        loss = torch.mean((output - target)**2) # Mean squared error (Don't use, poor performance on nonlinears)
        # loss = torch.mean(torch.abs(output - target)) # Mean absolute error
        
        """
        Backward Propogate our error through the model
        This works backwards, setting the gradient of each weight/bias in the model
        Gradients are set based on our loss
        """
        loss.backward()

        if not self.optimized:
            """Update each parameter in the network"""
            with torch.no_grad():
                for param in self.network.parameters():
                    if param == None:
                        continue
                    param -= self.alpha * param.grad
                    """Zero gradients to allow for continued propogation"""
                    param.grad.zero_()

        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
                
        return loss