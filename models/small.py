import torch
import torch.nn as nn
import torch.nn.functional as F

class smallNet(nn.Module) :
    
    def __init__(self) :
        super(smallNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,10,5)
        self.conv3 = nn.Conv2d(10,16,5)
        
        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024,5)
        
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        
    def forward(self, x) :
        
        batchSize = x.size(0)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(batchSize, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x