from pathlib import Path
import sys

import numpy as np
import tifffile as tiff

sys.path.append(str(Path(__file__).parents[1].resolve()))


# import optical_system_simulation
# import modules.optical_system_simulation
from modules.optical_system_simulation import OpticalSystemSimulation

img_nr = 3
path_img = Path(f'/Users/xx3662/Documents/Projekte_Daten/Marker_Synthesis/Data/Membrane-Nuclei/cycleGAN/binMembrane2nucSeg_td_v2_bRed/trainA_Raw/{img_nr}_prepro_scaled_membrane.tif')
path_spheroid_mask= Path(f'/Users/xx3662/Documents/Projekte_Daten/Marker_Synthesis/Data/Membrane-Nuclei/cycleGAN/binMembrane2nucSeg_td_v2_bRed/tmp_spheroid_mask/{img_nr}_prepro_scaled_dil.tif')
brightness_red_fuction = 'f3p'
brightness_red_factor = 600

img = tiff.imread(path_img).astype(np.float32)

oss = OpticalSystemSimulation()


params = {
    "path_output": None,
    "paths_psf": None,
    "path_spheroid_mask": str(path_spheroid_mask),
    "z_append": 0,
    "shape_des": img.shape,
    "brightness_red_fuction": brightness_red_fuction,
    "brightness_red_factor": brightness_red_factor,
    "gpu_conv": False,
}
oss.import_settings(params)


img_bred = oss.brightness_reduction(img, function=brightness_red_fuction, factor=brightness_red_factor)

print('Done')

dtype = np.uint16
max_val = np.iinfo(dtype).max
img_bred = np.clip(img_bred*max_val, 0, max_val).astype(dtype)
tiff.imwrite(path_img.parent / (path_img.stem + '_bRed' + path_img.suffix), img_bred)