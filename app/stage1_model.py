"""
stage1_model.py
===============
PyTorch CNN architecture for Stage 1 (Binary Cry Detection).
Expects MFCC input shape: (Batch, 1, 16, MFCC_Time_Frames)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage1CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input channel, 16 output channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 16 input channels, 32 output channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # 32 channels * 4 (height) * 40 (width) = 5120
        self.fc1 = nn.Linear(32 * 4 * 40, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x



# ── Factory ───────────────────────────────────────────────────────────────────
def build_stage1_model(device: torch.device):
    """
    Build and return the Stage 1 model instance. 
    Weights are NOT loaded here; the pipeline will load them via load_state_dict.
    """
    model = Stage1CNN().to(device)
    model.eval()
    return model