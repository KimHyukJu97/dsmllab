import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4868, 64)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(64, 8)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(8, 2)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x