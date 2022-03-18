import os
import numpy as np

import torch
import torch.nn as nn

from multiview_detector.models.tcn import TemporalConvNet

class DeepGlimpse(nn.Module):
    def __init__(self, num_inputs=100, in_dim=1, feat_dim=64) -> None:
        super().__init__()

        self.base = nn.Sequential(nn.Conv2d(in_dim, feat_dim, 3, padding=1), nn.ReLU(), nn.Conv2d(feat_dim, feat_dim, 1))
        self.glimpse = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.ReLU(), nn.Conv2d(feat_dim, feat_dim, 1))
        self.t_cnn = TemporalConvNet(num_inputs, feat_dim)
    
    def forward(self, x):
        pass