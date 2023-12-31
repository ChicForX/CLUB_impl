import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import TensorDataset, DataLoader

class S_Evaluator(nn.Module):
    def __init__(self, img_dim, dim_s):
        super(S_Evaluator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)

        num_features = 128 * (img_dim // 4) * (img_dim // 4)

        self.fc1 = nn.Linear(num_features, 512 * dim_s)
        self.bn3 = nn.BatchNorm1d(512 * dim_s)
        self.fc2 = nn.Linear(512 * dim_s, dim_s)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


