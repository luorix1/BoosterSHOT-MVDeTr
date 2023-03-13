import argparse

from multiview_detector.evaluation.TrackEval import trackeval

# This code is built on the TrackEval repo cited below

# @misc{luiten2020trackeval,
#   author =       {Jonathon Luiten, Arne Hoffhues},
#   title =        {TrackEval},
#   howpublished = {\url{https://github.com/JonathonLuiten/TrackEval}},
#   year =         {2020}
# }


def evaluationTracking_py(res_fpath, gt_fpath, dataset):
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config["DISPLAY_LESS_PROGRESS"] = False
    default_dataset_config = (
        trackeval.datasets.DeepingSourceTop2DBox.get_default_dataset_config()
    )
    default_metrics_config = {"METRICS": ["CLEAR"], "THRESHOLD": 0.5}
    config = {
        **default_eval_config,
        **default_dataset_config,
        **default_metrics_config,
    }  # Merge default configs
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs="+")
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == "True":
                    x = True
                elif args[setting] == "False":
                    x = False
                else:
                    raise Exception(
                        "Command line parameter " + setting + "must be True or False"
                    )
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == "SEQ_INFO":
                x = dict(zip(args[setting], [None] * len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {
        k: v for k, v in config.items() if k in default_dataset_config.keys()
    }
    metrics_config = {
        k: v for k, v in config.items() if k in default_metrics_config.keys()
    }

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.DeepingSourceTop2DBox(dataset_config)]
    metrics_list = []
    for metric in [
        trackeval.metrics.HOTA,
        trackeval.metrics.CLEAR,
        trackeval.metrics.Identity,
        trackeval.metrics.VACE,
    ]:
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception("No metrics selected for evaluation")
    res, msg = evaluator.evaluate(dataset_list, metrics_list)
    results = res["DeepingSourceTop2DBox"]["MPNTrack"]["COMBINED_SEQ"]["pedestrian"][
        "CLEAR"
    ]

    return results["CLR_F1"], results["MOTA"], results["MOTP"]


if __name__ == "__main__":
    f1, mota, motp = evaluationTracking_py(None, None, None)
    print(f"IDF1: {f1}, MOTA: {mota}, MOTP: {motp}")
