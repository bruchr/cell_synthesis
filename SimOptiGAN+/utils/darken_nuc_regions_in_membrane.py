import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage.util import random_noise

'''
Script to darkern nuclear regions in the membrane signal.
This is useful for improving the realism of simulated membrane signals, as this effect is also present in real membrane recordings.
'''

def darken_regions(path_membrane_img:Path, path_nuc_label:Path, offset:int=None) -> np.ndarray:

    img = tiff.imread(path_membrane_img)
    labels = tiff.imread(path_nuc_label)

    if img.shape != labels.shape:
        raise ValueError(f'Image dimensions must match, but are {img.shape} and {labels.shape}.')
    img = img.astype(np.float32)
    normal_distribution = np.random.normal(0, scale=20, size=img.shape)
    img += normal_distribution
    img = np.clip(img, 0, 255).astype(np.uint8)

    if offset is not None:
        labels = np.pad(labels, ((offset, 0), (0,0), (0,0)))[:-offset,...]

    labels_bin = labels != 0 # binarize labels
    img[labels_bin] = 0 # set intensity of nuclei labels to zero
    
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Darkens nuclei regions in simulated membrane signals.")
    parser.add_argument("path_membrane_img", type=str,
                        help="Image path of membrane signal.")
    parser.add_argument("path_nuc_label", type=str,
                        help="Image path of nuclei labels.")
    parser.add_argument("out_path", type=str,
                        help="File path where the resulting image is saved.")
    parser.add_argument("--offset", type=int, required=False, default=None,
                        help="Z-Offset of label image to membrane image.")
    args = parser.parse_args()


    path_membrane_img = Path(vars(args)["path_membrane_img"])
    path_nuc_label = Path(vars(args)["path_nuc_label"])
    out_path = Path(vars(args)["out_path"])
    offset = vars(args)["offset"]

    membrane_img_adapted = darken_regions(path_membrane_img, path_nuc_label, offset)
    tiff.imwrite(out_path, membrane_img_adapted)