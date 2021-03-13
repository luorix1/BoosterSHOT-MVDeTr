import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import random
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from multiview_detector.datasets import *
from multiview_detector.models.mvdetr import MVDeTr
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.utils.str2bool import str2bool
from multiview_detector.trainer import PerspectiveTrainer


def main(args):
    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
    else:
        print('No sys.gettrace')
        is_debug = False

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # deterministic
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    trans = T.Compose([T.Resize([1080 * 8 // args.img_reduce, 1920 * 8 // args.img_reduce]), T.ToTensor(),
                       T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])
    if 'wildtrack' in args.dataset:
        base = Wildtrack(os.path.expanduser('~/Data/Wildtrack'))
    elif 'multiviewx' in args.dataset:
        base = MultiviewX(os.path.expanduser('~/Data/MultiviewX'))
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    train_set = frameDataset(base, train=True, transform=trans, world_reduce=args.world_reduce,
                             img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                             img_kernel_size=args.img_kernel_size, semi_supervised=args.semi_supervised)
    test_set = frameDataset(base, train=False, transform=trans, world_reduce=args.world_reduce,
                            img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                            img_kernel_size=args.img_kernel_size)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    # logging
    if args.resume is None:
        logdir = f'logs/{args.dataset}/{"debug_" if is_debug else ""}{"SS_" if args.semi_supervised else ""}' \
                 f'{args.world_feat}_bottleneck{args.bottleneck_dim}_' \
                 f'mse{int(args.use_mse)}_alpha{args.alpha}_drop{args.dropout}_id{args.id_ratio}_' \
                 f'worldR{args.world_reduce}_imgR{args.img_reduce}_' \
                 f'worldK{args.world_kernel_size}_imgK{args.img_kernel_size}_' \
                 f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
        os.makedirs(logdir, exist_ok=True)
        copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    else:
        logdir = f'logs/{args.dataset}/{args.resume}'
    print(logdir)
    print('Settings:')
    print(vars(args))

    # model
    model = MVDeTr(train_set, args.arch, world_feat_arch=args.world_feat,
                   reduction=args.reduction, bottleneck_dim=args.bottleneck_dim,
                   outfeat_dim=args.outfeat_dim, droupout=args.dropout)

    # base_param_ids = set(map(id, model.base.parameters()))
    # new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
    # optimizer = optim.SGD([{'params': model.base.parameters(), 'lr': args.lr * 0.1},
    #                        {'params': new_params}, ],
    #                       lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam([{'params': model.base.parameters(), 'lr': args.lr * 0.1},
    #                         {'params': new_params}, ], weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = GradScaler()

    def warmup_lr_scheduler(epoch, warmup_epochs=2, step=args.step_size):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.1 ** int(np.log(epoch) / np.log(step))

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)

    trainer = PerspectiveTrainer(model, logdir, denormalize, args.cls_thres, args.alpha, args.use_mse, args.id_ratio)

    # draw curve
    x_epoch = []
    train_loss_s = []
    test_loss_s = []
    test_moda_s = []

    # learn
    res_fpath = os.path.join(logdir, 'test.txt')
    if args.resume is None:
        print('Testing...')
        trainer.test(0, test_loader, res_fpath, visualize=True)
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss = trainer.train(epoch, train_loader, optimizer, scaler, scheduler)
            print('Testing...')
            test_loss, moda = trainer.test(epoch, test_loader, res_fpath, visualize=True)

            # draw & save
            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            test_loss_s.append(test_loss)
            test_moda_s.append(moda)
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s, test_moda_s)
            torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
    else:
        model.load_state_dict(torch.load(f'logs/{args.dataset}/{args.resume}/MultiviewDetector.pth'))
        model.eval()
    print('Test loaded model...')
    trainer.test(None, test_loader, res_fpath, visualize=True)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--semi_supervised', type=str2bool, default=False)
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--cls_thres', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18', 'mobilenet'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size for training (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=2021, help='random seed (default: None)')
    parser.add_argument('--deterministic', type=str2bool, default=False)

    parser.add_argument('--world_feat', type=str, default='conv',
                        choices=['conv', 'trans', 'deform_conv', 'deform_trans'])
    parser.add_argument('--bottleneck_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--world_kernel_size', type=int, default=20)
    parser.add_argument('--img_kernel_size', type=int, default=10)
    parser.add_argument('--reduction', type=str, default=None)
    parser.add_argument('--use_multicam', type=bool, default=False)

    args = parser.parse_args()

    main(args)
