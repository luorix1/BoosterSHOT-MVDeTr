import os
import numpy as np
import torch
import torch.nn as nn

from multiview_detector.models.mvdetr import MVDeTr


class MVDeTrTracker(nn.Module):
    def __init__(self, dataset, arch='resnet18', z=0, world_feat_arch='conv', bottleneck_dim=128, outfeat_dim=64, dropout=0.5) -> None:
        super().__init__()

        self.Rimg_shape, self.Rworld_shape = dataset.Rimg_shape, dataset.Rworld_shape
        self.indexing = dataset.base.indexing
        self.img_reduce = dataset.img_reduce
        self.world_reduce = dataset.world_reduce
        self.num_cam = dataset.num_cam

        self.hg = MVDeTr(arch, z, world_feat_arch, bottleneck_dim, outfeat_dim, dropout)

    def forward(self, x, M):
        pass