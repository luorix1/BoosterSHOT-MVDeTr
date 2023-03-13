"""
@Author: Du Yunhao
@Filename: opts.py
@Contact: dyh_bupt@163.com
@Time: 2022/2/28 19:41
@Discription: opts
"""
import argparse
import json
from os.path import join

data = {
    "MOT17": {
        "val": [
            "MOT17-02-FRCNN",
            "MOT17-04-FRCNN",
            "MOT17-05-FRCNN",
            "MOT17-09-FRCNN",
            "MOT17-10-FRCNN",
            "MOT17-11-FRCNN",
            "MOT17-13-FRCNN",
        ],
        "test": ["data"],
    },
    "MOT20": {"test": ["MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08"]},
}


class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "dataset",
            type=str,
            help="MOT17 or MOT20",
        )
        self.parser.add_argument(
            "mode",
            type=str,
            help="val or test",
        )
        self.parser.add_argument(
            "--BoT",
            action="store_true",
            help="Replacing the original feature extractor with BoT",
        )
        self.parser.add_argument("--ECC", action="store_true", help="CMC model")
        self.parser.add_argument("--NSA", action="store_true", help="NSA Kalman filter")
        self.parser.add_argument(
            "--EMA", action="store_true", help="EMA feature updating mechanism"
        )
        self.parser.add_argument(
            "--MC",
            action="store_true",
            help="Matching with both appearance and motion cost",
        )
        self.parser.add_argument(
            "--woC",
            action="store_true",
            help="Replace the matching cascade with vanilla matching",
        )
        self.parser.add_argument(
            "--AFLink", action="store_true", help="Appearance-Free Link"
        )
        self.parser.add_argument(
            "--GSI", action="store_true", help="Gaussian-smoothed Interpolation"
        )
        self.parser.add_argument(
            "--dir_dataset", default="/workspace/MVDeTr_research/bev"
        )
        self.parser.add_argument(
            "--path_AFLink",
            default="/workspace/MVDeTr_research/multiview_detector/trackers/StrongSORT/data/AFLink_epoch20.pth",
        )
        self.parser.add_argument(
            "--dir_save",
            default="/workspace/MVDeTr_research/multiview_detector/trackers/StrongSORT/data",
        )
        self.parser.add_argument("--EMA_alpha", default=0.9)
        self.parser.add_argument("--MC_lambda", default=0.98)
        self.parser.add_argument(
            "--res_fpath",
            default="/workspace/MVDeTr_research/multiview_detector/evaluation/TrackEval/data/trackers/deeping_source/MOT17-train/MPNTrack/data/data.txt",
        )
        self.parser.add_argument(
            "--gt_fpath",
            default="/workspace/MVDeTr_research/multiview_detector/evaluation/TrackEval/data/gt/deeping_source/MOT17-train/data/gt/gt.txt",
        )

    def parse(self, args=""):
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        opt.min_confidence = 0.6
        opt.nms_max_overlap = 1.0
        opt.min_detection_height = 0
        opt.dir_dets = (
            "/workspace/MVDeTr_research/multiview_detector/trackers/StrongSORT/data"
        )
        if opt.BoT:
            opt.max_cosine_distance = 0.4
        else:
            opt.max_cosine_distance = 0.3
        if opt.MC:
            opt.max_cosine_distance += 0.05
        if opt.EMA:
            opt.nn_budget = 1
        else:
            opt.nn_budget = 100
        if opt.ECC:
            path_ECC = "/workspace/MVDeTr_research/multiview_detector/trackers/StrongSORT/data/{}_ECC_{}.json".format(
                opt.dataset, opt.mode
            )
            opt.ecc = json.load(open(path_ECC))
        opt.sequences = data[opt.dataset][opt.mode]
        return opt


opt = opts().parse()
