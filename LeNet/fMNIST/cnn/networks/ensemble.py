import torch 
import torch.nn as nn
from . import LeNet

'''
Note
    - You should use nn.ModuleList. 
    - Optimizer doesn't detect python list as parameters

'''


class LeNetEnsemble(nn.Module):
    def __init__(self, nClasses, ensemble=5, device='cuda'):
        super(LeNetEnsemble,self).__init__()
        self.nClasses = nClasses
        self.device = device
        self.models = nn.ModuleList(
                    [LeNet(self.nClasses).to(device) for _ in range(ensemble)]
                )
    
    def forward(self,x):
        output = torch.zeros([x.size(0), self.nClasses]).to(self.device)
        for model in self.models:
            output += model(x)
        return output


