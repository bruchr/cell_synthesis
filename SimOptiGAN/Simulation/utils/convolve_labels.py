import time

import numpy as np
from scipy.ndimage import zoom
import skimage.measure
import torch
from torch.nn.functional import conv3d

def convolve_label(label, psf, scale, t=0.25, sparse_labels=False, verbose=False):
    """
    Convolution of label images.
    Helpfull, if labels should match the size of nuclei in convolved nuclei image.
    Scale is necessary if image data was convolved in higher resolution than the output.
    
    Args:
        label: label image
        psf: psf image
        scale: needed if image convolution was performed in a different resolution than the output. scale = px_size_desired / px_size_sim_img
        t: threshold used to convert convolution result back to a binary label
        sparse_labels: if label is of sparse type (label=1 is assumed to be background and label=0 to be ignored)
        verbose: if True, additional information is printed
    Returns:
        Image, with just the biggest object left
    """
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    else:
        print('Conv: CPU is used. This may take a while.')
    
    if scale != 0 or scale != 1:
        label = zoom(label, zoom=scale, order=0, prefilter=False)
    if sparse_labels:
        label = label-1
    
    if verbose:
        print('Conv'+'-'*20)
        print('Conv: Input: Min/Max value in label: {} / {}. dtype: {}'.format(np.min(label), np.max(label), label.dtype))
        print('Conv: Shape of label: {}'.format(label.shape))

    label_new = np.zeros(np.shape(label))
    label_new_bin = np.zeros(np.shape(label), dtype=np.uint16)
    
    # Low and high padding in case of even shaped psf
    pad_l = np.floor(np.subtract(psf.shape, 1)/2).astype(int)
    pad_h = np.ceil(np.subtract(psf.shape, 1)/2).astype(int)
    
    psf = psf/psf.sum()
    psf = psf[None, None, ...]
    psf_c = torch.from_numpy(psf).to(device)
    
    props = skimage.measure.regionprops(label)
    for prop in props:
        # Convolution of each label on its own on a cropped image
        
        # Padding for influence range of psf
        bb_1 = np.stack([prop.bbox[0:3], prop.bbox[3:6]], axis=1)
        bb_1[:,0] = np.maximum(bb_1[:,0] - pad_h, 0)
        bb_1[:,1] = np.minimum(bb_1[:,1] + pad_l, label.shape)

        # Padding for same output size after convolution
        bb_2 = np.empty_like(bb_1)
        bb_2[:,0] = bb_1[:,0] - pad_l
        bb_2[:,1] = bb_1[:,1] + pad_h

        c_l_p = prop.coords - bb_2[:,0]
        shape_loc = bb_2[:,1] - bb_2[:,0]

        label_loc = np.zeros((shape_loc), dtype=np.float32)
        label_loc[tuple(c_l_p.T)] = 1.
        label_loc_c = torch.from_numpy(label_loc[None, None, ...]).to(device)
        
        label_loc = conv3d(label_loc_c, psf_c, padding='valid').cpu().numpy()[0,0,...]

        label_loc_bin = (label_loc > t) * prop.label

        c_l = np.stack(np.where(label_loc_bin), axis=1)
        c_g = c_l + bb_1[:,0]

        sl_l = tuple(c_l.T)
        sl_g = tuple(c_g.T)
        label_new_bin[sl_g] = np.where(label_loc[sl_l] > label_new[sl_g],
            label_loc_bin[sl_l], label_new_bin[sl_g])
        label_new[sl_g] = np.where(label_loc[sl_l] > label_new[sl_g],
            label_loc[sl_l], label_new[sl_g])

    print('Conv: Output: Min/Max value in label: {} / {}. dtype: {}'.format(np.min(label_new_bin), np.max(label_new_bin), label.dtype))

    if sparse_labels:
        label_new_bin += 1
    label_new = zoom(label_new, zoom=np.divide(1,scale))
    label_new_bin = zoom(label_new_bin, zoom=np.divide(1,scale), order=0, prefilter=False)

    return label_new.astype(np.float32), label_new_bin



if __name__=='__main__':
    import os
    import tifffile as tiff

    path = 'D:/Bruch/Projekte_und_Daten/Data_Synthesis/Simulation/Runs/Mario_A549_Mono_96h_Doxo_0uM_Ki/Run_4_worse_quality/labels_sparse'
    sparse_labels = True
    threshold = 0.4
    scale = (0.9999284/0.9002694, 0.5681821/0.1221902, 0.5681821/0.1221902)
    save_path = path + f'_conv_t{str(threshold).replace(".","d")}'
    path_psf = 'D:/Bruch/Projekte_und_Daten/Data_Synthesis/Simulation/Proc_Data/PSF/extraxted_psf_9d3zoom_32b_minCrop__scaled-0d5-0d5-0d5.tif'
    
    os.makedirs(save_path, exist_ok=True)

    for f_name in os.listdir(path):
        if f_name.endswith('.tif'):
            print(f_name)
            f_path = os.path.join(path, f_name)
            
            label = tiff.imread(f_path)
            s = time.time()
            _, label_bin = convolve_label(label, tiff.imread(path_psf), sparse_labels=sparse_labels, scale=scale, t=threshold, verbose=True)
            d = time.time() - s; print(f"Time needed for {f_name}: {d/60:.1f} min")

            tiff.imwrite(os.path.join(save_path, f_name.replace('.tif','_conv.tif')), label_bin)