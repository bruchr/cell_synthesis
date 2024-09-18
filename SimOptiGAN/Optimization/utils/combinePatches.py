import json
import os

import numpy as np
import tifffile as tiff

from utils.sliding_window import generateSlices


def combine_patches(path_results, im_nr, path_json, mode='half-cut', gen_ind_file=False):
    if not mode in ['naive','half-cut']:
        raise ValueError('mode must be either ''naive'' or ''half-cut'', but was ''{}'''.format(mode))

    if not path_json.endswith('.json'): path_json = os.path.join(path_json, 'slice_info_{}.json'.format(im_nr))
    with open(path_json, 'r') as json_file: json_dict = json.load(json_file)

    shape = json_dict['shape']
    p_size = json_dict['p_size']
    overlap = json_dict['overlap']

    slices, _ = generateSlices(shape, p_size, overlap)

    out = np.zeros(shape)
    if gen_ind_file:
        out_ind = np.zeros(shape, dtype=np.uint16)

    if not isinstance(overlap, (list, tuple)):
        overlap = (overlap, overlap, overlap) if len(shape) == 3 else (overlap, overlap)

    if mode=='half-cut' and np.all(np.asarray(overlap)<=0):
        raise ValueError('Overlap must be greater 0 if mode \'half-cut\' is used. Overlap={}'.format(overlap))

    if mode == 'half-cut':
        cut_point = np.asarray([
            [np.floor(p_size[ind]*overlap[ind]/2), np.ceil(p_size[ind]*overlap[ind]/2)] for ind in range(len(p_size))
            ]).astype(np.uint16)

    print('Combining patches...')
    for ind, sl in enumerate(slices):
        
        path_result = os.path.join(path_results, '{}_{}.tif'.format(im_nr, ind))
        if os.path.isfile(path_result):
            res = tiff.imread(path_result)
            
            if mode=='half-cut':
                sl_out, sl_res = __get_slices_halfcut(sl, p_size, cut_point, shape)
                if res.ndim == 2:
                    out[sl_out[0],sl_out[1],sl_out[2]] = res[sl_res[1],sl_res[2]]
                else:
                    out[sl_out[0],sl_out[1],sl_out[2]] = res[sl_res[0],sl_res[1],sl_res[2]]
                if gen_ind_file:
                    out_ind[sl_out[0],sl_out[1],sl_out[2]] = ind
            else:
                out[sl[0],sl[1],sl[2]] = res
                if gen_ind_file:
                    out_ind[sl[0],sl[1],sl[2]] = ind
                # out[sl[0],sl[1],sl[2]] = np.maximum(out[sl[0],sl[1],sl[2]], res)
                # out[sl[0],sl[1],sl[2]] += res
                # out_counter[sl[0],sl[1],sl[2]] += 1


    # out[out_counter!=0] = out[out_counter!=0]/out_counter[out_counter!=0]
    path_output = os.path.join(path_results, 'combined_{}_{}.tif'.format(mode, im_nr))
    tiff.imwrite(path_output, out.astype(np.float32))
    if gen_ind_file:
        tiff.imwrite(path_output.replace('.tif','_ind.tif'), out_ind)
    print('Saved results in: {}'.format(path_output))


def __get_slices_halfcut(sl_in, p_size, cut_point, im_shape):
    '''Calculate the slice indices for the halfcut variant'''
    sl_out = list(sl_in)
    sl_res = [slice(0, val) for val in p_size]

    for ind in range(len(sl_in)):
        if sl_in[ind].start != 0: # if not at boarder start
            sl_out[ind] = slice(sl_out[ind].start + cut_point[ind,0], sl_out[ind].stop)
            sl_res[ind] = slice(sl_res[ind].start + cut_point[ind,0], sl_res[ind].stop)
        if sl_in[ind].stop != im_shape[ind]: # X, if not at boarder end
            sl_out[ind] = slice(sl_out[ind].start, sl_out[ind].stop-cut_point[ind,1])
            sl_res[ind] = slice(sl_res[ind].start, sl_res[ind].stop-cut_point[ind,1])

    return sl_out, sl_res



if __name__ == '__main__':
    
    path_results = 'D:/Bruch/CycleGAN_TMP/cyclegan_own/checkpoints/td_2dTrain_F3_OnlyPoisson_crop_cGAN_202208111818_7000/results_data_2dTrain_F3_OnlyPoisson_20220331_crop_switch/AtoB'
    path_json = path_results
    combine_patches(path_results, 0, path_json, mode='half-cut')