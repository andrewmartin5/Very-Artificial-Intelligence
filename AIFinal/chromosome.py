import torch
import torch.nn as nn

class Chromosome(nn.Module):
    # TODO would be cool to make it more expandable in the future
    def __init__(self, numInputs, numOutputs, firstLayerSize=8, secondLayerSize=4):
        super(Chromosome, self).__init__()
        self.layer1 = nn.Linear(numInputs, firstLayerSize)
        self.layer2 = nn.Linear(firstLayerSize, secondLayerSize)
        self.layer3 = nn.Linear(secondLayerSize, numOutputs)
        self.activation = nn.ReLU() # the activation function for the hidden layers

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = torch.softmax(self.layer3(x), dim=-1) # softmax is needed to convert output layer to probabilities
        return x
    

