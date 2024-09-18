import os
import sys
import numpy as np
import tifffile as tiff
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from spheroid_simulator import SpheroidSimulator

json_file = 'C:/Users/xx3662/Desktop/Projekte/Paper_Synthesis_Data/Simulation/Runs/F3_OnlyPoisson_20220331_binLabel/params.json'


simulator = SpheroidSimulator()
simulator.import_settings_json(json_file)

simulator.optical_system_simulation.import_settings(simulator.optical_system_params)
simulator.optical_system_simulation.z_append = 0

out_path = simulator.path_output

for ind, path_spheroid_mask in enumerate(simulator.path_spheroid_mask_list):

    function = simulator.optical_system_simulation.brightness_red_fuction
    # minimum = simulator.optical_system_simulation.min_brightness_perc
    factor = simulator.optical_system_simulation.brightness_red_factor

    simulator.optical_system_simulation.path_spheroid_mask = path_spheroid_mask
    
    label_path = os.path.join(simulator.path_output, simulator.path_out_folders[1], "{:03d}_{:03d}_final_label.tif".format(ind, 1))
    label = tiff.imread(label_path)
    label[label!=0] = 255
    label = label.astype(np.uint8)

    label_b_red = simulator.optical_system_simulation.brightness_reduction(label, function=function, factor=factor)

    path_out = os.path.join(simulator.path_output, 'labels_bRed', "{:03d}_{:03d}_final_label_bRed.tif".format(ind, 1))
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    tiff.imwrite(path_out, label_b_red)

    label_noise = (label_b_red*(230/255)).astype(np.uint8)
    label_noise = simulator.camera_acquisition_simulation.add_gauss_noise(label_noise, 0, 10)
    path_out_noise = os.path.join(simulator.path_output, 'labels_bRed_noise', "{:03d}_{:03d}_final_label_bRed_noise.tif".format(ind, 1))
    os.makedirs(os.path.dirname(path_out_noise), exist_ok=True)
    tiff.imwrite(path_out_noise, label_noise)