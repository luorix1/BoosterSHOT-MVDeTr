import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import DataLoader

from multiview_detector.datasets import *

def bev_visualize(args):
    if 'wildtrack' in args.dataset:
        base = Wildtrack(os.path.expanduser('/workspace/Data/Wildtrack'))
    elif 'multiviewx' in args.dataset:
        base = MultiviewX(os.path.expanduser('/workspace/Data/MultiviewX'))
    else:
        raise Exception('must choose from [retail, wildtrack, multiviewx]')
    
    if args.subset == 'test':
        dataset = frameDataset(base, train=False, world_reduce=args.world_reduce,
                                img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                                img_kernel_size=args.img_kernel_size, task='tracking')
    else:
        raise Exception('only test set evaluation is supported at the moment.')

    gtRaw = np.loadtxt(args.gt_fpath)
    detRaw = np.loadtxt(args.res_fpath)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                             pin_memory=True, worker_init_fn=seed_worker)

    for batch_idx, (data, world_gt, imgs_gt, affine_mats, frame) in enumerate(dataloader):
        print(f'Processing batch {batch_idx + 1}')

        if batch_idx >= 3:
            break
        else:
            gt_t = gtRaw[gtRaw[:,0] == batch_idx + 1, :]
            det_t = detRaw[detRaw[:,0] == batch_idx + 1, :]
            
            shape = world_gt['heatmap'].squeeze().numpy().shape
            gt_map = np.zeros(shape)
            det_map = np.zeros(shape)
            for gtRow, detRow in zip(gt_t, det_t):
                # For now, don't apply Hungarian Algorithm and just show the PIDs for each (will fix later)
                gt_x = (gtRow[2] + gtRow[4]) / 2 / args.world_reduce
                gt_y = (gtRow[3] + gtRow[5]) / 2 / args.world_reduce
                gt_map[int(gt_x)][int(gt_y)] = 1
                
                det_x = (detRow[2] + detRow[4]) / 2 / args.world_reduce
                det_y = (detRow[3] + detRow[5]) / 2 / args.world_reduce
                det_map[int(det_x)][int(det_y)] = 1

                fig = plt.figure()
                subplt1 = fig.add_subplot(121)
                subplt2 = fig.add_subplot(122)
                subplt1.set_xticks([])
                subplt1.set_yticks([])
                subplt2.set_xticks([])
                subplt2.set_yticks([])

                subplt1.imshow(det_map)
                subplt2.imshow(world_gt['heatmap'].squeeze().numpy())

                plt.savefig(f'/workspace/MVDeTr_research/output/diagram_{batch_idx + 1}.png')
                



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gt_fpath', help='Path to file contatining GTs for MOT')
    parser.add_argument('--res_fpath', help='Path to file contatining results for MOT')
    parser.add_argument('-d', '--dataset', default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('--subset', default='test', choices=['all', 'train', 'test'])
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)

    args = parser.parse_args()

    bev_visualize(args)