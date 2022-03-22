import argparse
import os
import numpy as np


def reformat_file(args):
    file_path = os.path.join(args.file_path, 'test.txt')
    dets = np.loadtxt(file_path)
    dets[:,0] = (dets[:,0] - dets[:,0].min()) // 5 + 1
    dets = np.concatenate((dets[:,0][:,None], -np.ones((dets.shape[0], 1)), (dets[:,1:] - 1), (dets[:,1:] + 1), np.ones((dets.shape[0],1)), -np.ones((dets.shape[0],3))), axis=1)
    np.savetxt(os.path.join(args.file_path, 'det.txt'), dets, '%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path')
    args = parser.parse_args()
    reformat_file(args)