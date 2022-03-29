import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random

import torch
from torch.utils.data import DataLoader

from multiview_detector.utils.image_utils import draw_umich_gaussian
from multiview_detector.datasets import *


def main(args):
    # dataset is set to Wildtrack
    base = Wildtrack(os.path.expanduser('/workspace/Data/Wildtrack'))
    
    test_set = frameDataset(base, train=False, world_reduce=args.world_reduce,
                            img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                            img_kernel_size=args.img_kernel_size)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4,
                             pin_memory=True, worker_init_fn=seed_worker)

    results = np.loadtxt(args.res_fpath)
    
    for batch_idx, (_, _, _, _, frame) in enumerate(test_loader):
        H, W = test_set.Rworld_shape
        heatmap = np.zeros([1, H, W], dtype=np.float32)

        res_map_grid = results[results[:, 0] == (frame.detach().cpu().numpy() - 1800) // 5 + 1, 1:]

        for result in res_map_grid:
            ct = np.array([(result[2] + result[4]) / 2 / test_set.world_reduce, (result[3] + result[5]) / test_set.world_reduce], dtype=np.float32)
            if 0 <= ct[0] < W and 0 <= ct[1] < H:
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(heatmap[0], ct_int, test_set.world_kernel_size / test_set.world_reduce)

        print(f'Running batch {batch_idx + 1}')
        
        plt.imsave(os.path.join('/workspace/MVDeTr_research/bev', f'{(batch_idx + 1):08}.png'), heatmap[0])

    pass        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--res_fpath', help='Absolute path of parent directory', default='/workspace/MVDeTr_research/multiview_detector/evaluation/TrackEval/data/trackers/deeping_source/MOT17-train/MPNTrack/data/data.txt')

    parser.add_argument('--bottleneck_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)
    
    args = parser.parse_args()
    main(args)