import os
import shutil
from sysconfig import get_path
from types import NoneType

from matplotlib import pyplot as plt
import numpy as np
import tifffile as tiff

'''Adjust the brightness of all prototypes in a folder.
Info file is copied and appended with information on new max value.'''

def get_brightness(dir_path:str) -> list:
    path_list = get_path_list(dir_path)
    brightness = []
    for path in path_list:
        img = tiff.imread(path)
        brightness.append(np.quantile(img.ravel(), 0.99))
    return brightness

def get_path_list(data_path:str) -> list:
    file_list = []
    for file_name in os.listdir(data_path):
        if file_name.endswith('.tif') and 'label' not in file_name:
            file_list.append(os.path.join(data_path, file_name))
    return file_list

def adjust_brightness(dir_path:str, new_max:int, out_path:str) -> None:
    os.makedirs(out_path, exist_ok=False)
    copy_info(dir_path, out_path)
    
    path_list = get_path_list(dir_path)
    for ind, path in enumerate(path_list):
        img = tiff.imread(path)
        img = np.clip(np.round(img/new_max*255), 0, 255).astype(np.uint8)
        
        new_path = os.path.join(out_path, os.path.basename(path))
        tiff.imwrite(new_path, img, imagej=True)
        copy_label(path, new_path)
        write_info(out_path, os.path.basename(path), new_max, new_line=ind==0)

def copy_label(img_path_old:str, img_path_new:str) -> None:
    label_path_old = img_path_old.replace('.tif','_label.tif')
    label_path_new = img_path_new.replace('.tif','_label.tif')
    shutil.copy(label_path_old, label_path_new)

def write_info(dir_path:str, file_name:str, new_max:int, new_line=False) -> None:
    with open(os.path.join(dir_path, 'info.txt'), mode='a') as txt_file:
        if new_line: txt_file.write('\n\n')
        txt_file.write('New max: {:3d} for file: {}\n'.format(new_max, file_name))

def copy_info(dir_path_old:str, dir_path_new:str) -> None:
    path_old = os.path.join(dir_path_old, 'info.txt')
    path_new = os.path.join(dir_path_new, 'info.txt')
    if os.path.exists(path_old):
        shutil.copy(path_old, path_new)


if __name__ == '__main__':

    data_path = '/Users/xx3662/Documents/Projekte_Daten/Paper_Synthesis_Data/Simulation/Phantoms/MDA_HighRes_AugmentedV2'
    # data_path = '/Users/xx3662/Documents/Projekte_Daten/Paper_Synthesis_Data/Simulation/Phantoms/COX002_prototypes_v1_augmented_v1'
    out_path = data_path + '_adjBright'

    brightness = get_brightness(data_path)
    print(np.mean(brightness))

    # plt.bar(range(len(brightness)), brightness)
    # plt.show()
    
    adjust_brightness(data_path, new_max=200, out_path=out_path)