import torch.nn as nn
import torch.nn.functional as F

class SmallMLP(nn.Module):
    def __init__(self, input_dim, hidden=256, num_classes=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, num_classes)
    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))
        logits = self.out(feat)
        return logits
