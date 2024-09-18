import json
import os
import random

import numpy as np
import tifffile as tiff

from utils.sliding_window import generateSlices


def create_patches(path_img, path_sim, path_output, im_nr, mode='train', path_label=None, p_size=[1,500,500], overlap=0.25, no_bg=False):

    if not mode in ['train','inference']:
        raise ValueError('mode must be either \'train\' or \'inference\', but was \'{}\''.format(mode))
    if no_bg and path_label is None:
        raise Exception('If no_gb is selected, the path of the labelmap is required!')
    
    # Create output paths and make dirs
    path_outputA = os.path.join(path_output, mode+'A')
    path_outputB = os.path.join(path_output, mode+'B')
    if path_label is not None:
        path_outputB_label = os.path.join(path_output, mode+'_labelB')
        os.makedirs(path_outputB_label, exist_ok=True)
    os.makedirs(path_outputA, exist_ok=True)
    os.makedirs(path_outputB, exist_ok=True)

    # Read the tiffs
    orig = tiff.imread(path_img)
    sim = tiff.imread(path_sim)
    if path_label is not None:
        label = tiff.imread(path_label)
    else:
        label = None

    # Smallest shape is used for creation of patches
    shape = np.minimum(orig.shape,sim.shape)
    if p_size is None:
        # Use the full image slice
        p_size = [1, int(shape[1]), int(shape[2])]
    
    slices, _ = generateSlices(shape, p_size, overlap)
    
    # Save the info as json
    json_dict = {'shape': shape.tolist(), 'p_size': p_size, 'overlap': overlap, 'no_bg': no_bg,
    'path_img': path_img, 'path_sim': path_sim}
    if path_label is not None:
        json_dict['path_label'] = path_label
    
    json_dict['path_out'] = path_outputA
    with open(os.path.join(path_outputA, 'slice_info_{}.json'.format(im_nr)), 'w') as json_file: json.dump(json_dict, json_file, indent=4)
    json_dict['path_out'] = path_outputB
    with open(os.path.join(path_outputB, 'slice_info_{}.json'.format(im_nr)), 'w') as json_file: json.dump(json_dict, json_file, indent=4)
    if path_label is not None:
        with open(os.path.join(path_outputB_label, 'slice_info.json'), 'w') as json_file: json.dump(json_dict, json_file, indent=4)

    print('Creating patches...')

    # Slice the image with the calculated slices
    counter = 0
    for ind, sl in enumerate(slices):
        orig_slice = np.squeeze(orig[sl])
        sim_slice = np.squeeze(sim[sl])
        if label is not None: label_slice = np.squeeze(label[sl])

        if not no_bg or np.any(label_slice) or random.random() >= no_bg:
            tiff.imwrite(os.path.join(path_outputA, '{}_{}.tif'.format(im_nr, ind)), orig_slice)
            tiff.imwrite(os.path.join(path_outputB, '{}_{}.tif'.format(im_nr, ind)), sim_slice)
            counter +=1
            if path_label is not None:
                tiff.imwrite(os.path.join(path_outputB_label, '{}_{}.tif'.format(im_nr, ind)), label_slice)

    print('Created {} patches from {} and {} in:\n{}'.format(counter, os.path.basename(path_img), os.path.basename(path_sim), path_outputA))


if __name__ == '__main__':

    path_img =    'D:/Bruch/CycleGAN/data_2dTrain_F3_OnlyPoisson_20220331_crop/trainA/1_no additive_DAPI_grey.tif'
    path_sim =    'D:/Bruch/CycleGAN/data_2dTrain_F3_OnlyPoisson_20220331_crop/trainB/000_001_final.tif'
    path_output = 'D:/Bruch/CycleGAN_TMP/data_2dTrain_F3_OnlyPoisson_20220331_crop'
    create_patches(path_img, path_sim, path_output, im_nr=0, path_label=None,
        mode='inference', p_size=[1,128,128], overlap=[0,0.5,0.5], no_bg=False)