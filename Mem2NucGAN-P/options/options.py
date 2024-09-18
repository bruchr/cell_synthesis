import argparse
from datetime import datetime
from pathlib import Path
import time

import hjson


def options_parser():
    """Create a Parser for the options being parsed"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str, help="Path to the settings file if not given settings.json from root directory is used")
    parser.add_argument("--experiment", type=str, help="Name of the experiment")
    parser.add_argument("--no_timestamp", action="store_true", help="Do not append time stamp to model name", default=False)
    parser.add_argument("--dataset_folder", type=str, help="Path to the dataset")
    parser.add_argument("--output_folder", type=str, help="Path to store the results")
    parser.add_argument("--batch_size", type=int, help="# of images per batch")
    parser.add_argument("--mode", type=str,  help="mode train: training, inf: inference")
    parser.add_argument("--learning_rate_fix", type=int, help="Number of epochs with fixed learning rate")
    parser.add_argument("--learning_rate_decay", type=int, help="Number of epochs with learning rate decay")
    parser.add_argument("--time_limit", type=int, help="Time limit in s", default=0)
    parser.add_argument("--resume_training", type=int, help="Resume the training of a saved model. 0 or 1.")
    parser.add_argument("--inf_patch_size", type=int, help="Patch size used during inference. If not used, no patching is performed", nargs='+')
    parser.add_argument("--inf_patch_overlap", type=float, help="Patch overlap used during inference given in percent", nargs='+')

    # Get arguments from --settings and overwrite all arguments not parsed (None)
    args = parser.parse_args()
    if args.settings is None:
        args.settings = "settings.hjson"

    settings_path = Path(args.settings)
    with open(settings_path) as infile:
        opt = hjson.load(infile)

    for key, value in vars(args).items():
        if value is not None:
            if key == 'inf_patch_overlap' and len(value) == 1:
                value = value[0]
            opt[key] = value
            print(key, value)

    # Add timestamp to experiment for training
    if opt["mode"] == "train" and not opt["no_timestamp"]:
        opt["experiment"] = opt["experiment"] + "_" + datetime.now().strftime('%Y%m%d%H%M')

    opt["time_start"] = time.time()

    return opt
