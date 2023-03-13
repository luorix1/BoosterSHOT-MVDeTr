import argparse
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
from ptflops import get_model_complexity_info

from multiview_detector.datasets import *
from multiview_detector.models.attnchannelcutoff import AttnChannelCutoff
from multiview_detector.models.mvdetr import MVDeTr
from multiview_detector.utils.image_utils import img_color_denormalize


def main(args):
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose(
        [
            T.Resize([720, 1280]),
            T.ToTensor(),
            normalize,
        ]
    )

    base = Wildtrack(os.path.expanduser("/workspace/Data/Wildtrack_old"))
    test_set = frameDataset(base, train=False)

    with torch.cuda.device(0):
        if args.model == "BoosterSHOT":
            net = AttnChannelCutoff(
                dataset=test_set,
                arch=args.arch,
                world_feat_arch="conv",
                depth_scales=args.depth_scales,
                topk=4,
            ).cuda()
        elif args.model == "MVDeTr":
            net = MVDeTr(
                dataset=test_set, arch=args.arch, z=0, world_feat_arch="deform_trans"
            ).cuda()
        else:
            raise Exception("This model is not supported.")
        macs, params = get_model_complexity_info(
            net,
            (7, 3, 720, 1280),
            as_strings=False,
            print_per_layer_stat=True,
            verbose=True,
        )
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print("{:<30}  {:<8}".format("GFLOPs: ", 2 * macs / 1000 / 1000 / 1000))
        print("{:<30}  {:<8}".format("Number of parameters: ", params))

        from datetime import datetime as dt

        time_list = []
        for _ in range(20):
            test_input = torch.randn([1, 7, 3, 720, 1280]).cuda()

            start = dt.now()
            _ = net(test_input)
            running_secs = (dt.now() - start).total_seconds()
            print(running_secs)
            time_list.append(running_secs)

        print(
            f"Average time: {np.array(time_list).mean()}, Std: {np.array(time_list).std()}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count the computational cost of a model"
    )
    parser.add_argument("--model", choices=["MVDeTr", "BoosterSHOT"])
    parser.add_argument("--depth_scales", type=int, default=4)
    # parser.add_argument('--topk', default=None)
    parser.add_argument(
        "--arch", default="resnet18", choices=["resnet18", "vgg11", "mobilenet"]
    )

    args = parser.parse_args()

    main(args)

# FLOPs = 2 * MACs
# FLOPs = Floating point operations
# MACs = Multiply–accumulate operations
