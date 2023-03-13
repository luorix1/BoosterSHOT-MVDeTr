import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from multiview_detector.datasets import *
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.models.attnchannelcutoff import AttnChannelCutoff
from multiview_detector.models.cbamshot import CBAMSHOT
from multiview_detector.models.cgshot import CGSHOT
from multiview_detector.models.mvdetr import MVDeTr
from multiview_detector.models.shot import SHOT
from multiview_detector.utils.decode import mvdet_decode
from multiview_detector.utils.nms import nms


def main(args):
    # dataset
    if "wildtrack_hard" in args.dataset:
        base = Wildtrack_hard(os.path.expanduser("/workspace/Data/Wildtrack"))
    elif "wildtrack" in args.dataset:
        base = Wildtrack(os.path.expanduser("/workspace/Data/Wildtrack"))
    elif "multiviewx_40" in args.dataset:
        base = MultiviewX_40(os.path.expanduser("/workspace/Data/MultiviewX"))
    elif "multiviewx" in args.dataset:
        base = MultiviewX(os.path.expanduser("/workspace/Data/MultiviewX"))
    else:
        raise Exception(
            "must choose from [wildtrack, wildtrack_hard, multiviewx, multiviewx_hard]"
        )

    test_set = frameDataset(
        base,
        train=False,
        world_reduce=args.world_reduce,
        img_reduce=args.img_reduce,
        world_kernel_size=args.world_kernel_size,
        img_kernel_size=args.img_kernel_size,
        task="detection",
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    if args.model == "MVDeTr":
        model = MVDeTr(
            test_set,
            args.arch,
            world_feat_arch=args.world_feat,
            bottleneck_dim=args.bottleneck_dim,
            outfeat_dim=args.outfeat_dim,
            dropout=0,
        ).cuda()
    elif args.model == "AttnChannelCutoff":
        model = AttnChannelCutoff(
            test_set,
            args.arch,
            world_feat_arch=args.world_feat,
            bottleneck_dim=args.bottleneck_dim,
            outfeat_dim=args.outfeat_dim,
            dropout=0,
            depth_scales=args.depth_scales,
        ).cuda()
    elif args.model == "CBAMSHOT":
        model = CBAMSHOT(
            test_set,
            args.arch,
            world_feat_arch=args.world_feat,
            bottleneck_dim=args.bottleneck_dim,
            outfeat_dim=args.outfeat_dim,
            dropout=0,
            depth_scales=args.depth_scales,
        ).cuda()
    elif args.model == "CGSHOT":
        model = CGSHOT(
            test_set,
            args.arch,
            world_feat_arch=args.world_feat,
            bottleneck_dim=args.bottleneck_dim,
            outfeat_dim=args.outfeat_dim,
            dropout=0,
            depth_scales=args.depth_scales,
        ).cuda()
    elif args.model == "SHOT":
        model = SHOT(
            test_set,
            args.arch,
            world_feat_arch=args.world_feat,
            bottleneck_dim=args.bottleneck_dim,
            outfeat_dim=args.outfeat_dim,
            dropout=0,
            depth_scales=args.depth_scales,
        ).cuda()
    else:
        raise Exception("The selected model is not supported.")

    ckpt = torch.load(os.path.join(args.ckpt_dir, "MultiviewDetector.pth"))
    model.load_state_dict(ckpt)

    model.eval()
    res_list = []
    for batch_idx, (data, _, imgs_gt, affine_mats, frame) in enumerate(test_loader):
        print("Processing batch ", batch_idx)
        B, N = imgs_gt["heatmap"].shape[:2]
        data = data.cuda()
        for key in imgs_gt.keys():
            imgs_gt[key] = imgs_gt[key].view([B * N] + list(imgs_gt[key].shape)[2:])
        # with autocast():
        with torch.no_grad():
            (world_heatmap, world_offset), (_, _, _) = model(
                data, affine_mats, args.ckpt_dir, False
            )

        os.makedirs(os.path.join(args.ckpt_dir, args.dataset), exist_ok=True)
        res_fpath = os.path.join(args.ckpt_dir, args.dataset, "test.txt")

        if res_fpath is not None:
            xys = mvdet_decode(
                torch.sigmoid(world_heatmap.detach().cpu()),
                world_offset.detach().cpu(),
                reduce=test_loader.dataset.world_reduce,
            )
            grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
            if test_loader.dataset.base.indexing == "xy":
                positions = grid_xy
            else:
                positions = grid_xy[:, :, [1, 0]]

            for b in range(B):
                ids = scores[b].squeeze() > args.cls_thres
                pos, s = positions[b, ids], scores[b, ids, 0]
                res = torch.cat([torch.ones([len(s), 1]) * frame[b], pos], dim=1)
                ids, count = nms(pos, s, 20, np.inf)
                res = torch.cat(
                    [torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1
                )
                res_list.append(res)

    if res_fpath is not None:
        res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
        np.savetxt(res_fpath, res_list, "%d")

        recall, precision, moda, modp = evaluate(
            "detection",
            os.path.abspath(res_fpath),
            os.path.abspath(test_loader.dataset.gt_fpath),
            test_loader.dataset.base.__name__,
        )
        print(
            f"moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["CBAMSHOT", "CGSHOT", "MVDeTr", "SHOT", "AttnChannelCutoff"]
    )
    parser.add_argument("--cls_thres", type=float, default=0.6)
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        choices=["vgg11", "resnet18", "mobilenet"],
    )
    parser.add_argument("--depth_scales", type=int, default=4)
    parser.add_argument(
        "--world_feat",
        type=str,
        default="conv",
        choices=["conv", "trans", "deform_conv", "deform_trans", "aio"],
    )
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--outfeat_dim", type=int, default=0)
    parser.add_argument("--world_reduce", type=int, default=4)
    parser.add_argument("--world_kernel_size", type=int, default=10)
    parser.add_argument("--img_reduce", type=int, default=12)
    parser.add_argument("--img_kernel_size", type=int, default=10)
    parser.add_argument("--ckpt_dir")
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["wildtrack", "wildtrack_hard", "multiviewx", "multiviewx_40"],
    )

    args = parser.parse_args()

    main(args)
