import torch
import torch.nn as nn

class LeNet(nn.Module):

    def __init__(self,nClasses):
        super(LeNet,self).__init__()

        # Build the architecture
        self.conv1 = nn.Sequential(
                    # Lenet's first conv layer is 3x32x32, squeeze color
                    # channels into 1 and pad 2
                    nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2,stride=2)
                )

        # 16 @ 10x10
        self.conv2 = nn.Sequential(
                    nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)                
                )

        self.fc1 = nn.Sequential(
                    nn.Linear(16*5*5, 120),
                    nn.ReLU(),
                )
        self.fc2 = nn.Sequential(
                    nn.Linear(120,84),
                    nn.ReLU()
                )
        
        self.classifier = nn.Linear(84,nClasses)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 16*5*5)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.classifier(x)
        return logits


        


