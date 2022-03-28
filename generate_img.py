import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random

import torch
from torch.utils.data import DataLoader

from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.datasets import *


def main(args):
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # dataset is set to MultiviewX
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

    # create directory
    os.makedirs('/workspace/MVDeTr_research/imgs', exist_ok=True)
    
    for batch_idx, (data, _, _, _, _) in enumerate(test_loader):
        fig = plt.figure(tight_layout=True)
        
        imgs = data.squeeze(0)

        subplt1 = fig.add_subplot(421)
        subplt2 = fig.add_subplot(422)
        subplt3 = fig.add_subplot(423)
        subplt4 = fig.add_subplot(424)
        subplt5 = fig.add_subplot(425)
        subplt6 = fig.add_subplot(426)
        subplt7 = fig.add_subplot(427)
        subplt1.set_xticks([])
        subplt1.set_yticks([])
        subplt2.set_xticks([])
        subplt2.set_yticks([])
        subplt3.set_xticks([])
        subplt3.set_yticks([])
        subplt4.set_xticks([])
        subplt4.set_yticks([])
        subplt5.set_xticks([])
        subplt5.set_yticks([])
        subplt6.set_xticks([])
        subplt6.set_yticks([])
        subplt7.set_xticks([])
        subplt7.set_yticks([])

        subplt1.imshow(denormalize(imgs[0]).detach().cpu().numpy().squeeze().transpose([1, 2, 0]))
        subplt2.imshow(denormalize(imgs[1]).detach().cpu().numpy().squeeze().transpose([1, 2, 0]))
        subplt3.imshow(denormalize(imgs[2]).detach().cpu().numpy().squeeze().transpose([1, 2, 0]))
        subplt4.imshow(denormalize(imgs[3]).detach().cpu().numpy().squeeze().transpose([1, 2, 0]))
        subplt5.imshow(denormalize(imgs[4]).detach().cpu().numpy().squeeze().transpose([1, 2, 0]))
        subplt6.imshow(denormalize(imgs[5]).detach().cpu().numpy().squeeze().transpose([1, 2, 0]))
        subplt7.imshow(denormalize(imgs[6]).detach().cpu().numpy().squeeze().transpose([1, 2, 0]))
        
        plt.subplots_adjust(wspace=0.1, hspace=0)
        plt.tight_layout()
        plt.savefig(f'/workspace/MVDeTr_research/imgs/{(batch_idx + 1):08}.png', bbox_inches='tight')

    pass        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--bottleneck_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)
    
    args = parser.parse_args()
    main(args)