import torch
import torch.nn as nn
import torch.nn.functional as F

class mediumNet(nn.Module) :
    
    def __init__(self) :
        super(mediumNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3,6,3)
        self.batchConv1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6,10,3)
        self.batchConv2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10,16,3)
        self.batchConv3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16,32,3)
        self.batchConv4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32,48,3)
        self.batchConv5 = nn.BatchNorm2d(48)

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(768,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,5)
        
        self.dropout = nn.Dropout(p=0.5)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x) :
        
        batchSize = x.size(0)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))

        x = x.view(batchSize, -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x