import torch.nn as nn

class Network(nn.Module):
    
    def __init__(self, observationSize, actionSize, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)