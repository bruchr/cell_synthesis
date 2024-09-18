import tifffile as tiff
import numpy as np
import os
from scipy.ndimage import generate_binary_structure
from skimage.transform import rescale
from random import randint
import sys
sys.path.append('D:/Bruch/Prototype_Simulation_Method/Pipeline')
from modules.phantom_image_generator import PhantomImageGenerator as PIG

path_mask = 'D:/Bruch/Prototype_Simulation_Method/data/Organoids_Mario/masks/spheroid_mask_0.tif'
path_nuclei_folder = 'D:/Bruch/Prototype_Simulation_Method/data/Processing_Files/extracted_phantoms'
path_nuclei_labels = [os.path.join(path_nuclei_folder, path) for path in os.listdir(path_nuclei_folder) if '_label.tif' in path]
path_output = path_mask.replace('.tif', '_lumen.tif')

max_overlap = 0
max_lumen_num = 25
dilation_structure = generate_binary_structure(3, 1)
scale = np.multiply((0.3, 0.125, 0.125),2)

mask = tiff.imread(path_mask)
mask[mask>0] = 1
mask_label = np.copy(mask)
mask_lumen = np.zeros_like(mask, dtype=np.bool)


args = np.argwhere(mask)
min_z = args[:, 0].min()
max_z = args[:, 0].max()
min_y = args[:, 1].min()
max_y = args[:, 1].max()
min_x = args[:, 2].min()
max_x = args[:, 2].max()

lumen_num = 2

while True:
    # create ellipsoids or use them from nuclei and scale them up

    # place them randomly inside the mask and set the mask to background
    r_ind = np.random.randint(0, len(path_nuclei_labels)-1)
    mask_object = rescale(tiff.imread(path_nuclei_labels[r_ind]), scale, order=0, channel_axis=None, preserve_range=True, anti_aliasing=False)
    mask_object[mask_object > 0] = 1
    mask_object_dil = PIG.get_dilated_phantom(mask_object, dilation_structure, min_dist=6)

    props = PIG.get_region_props(mask_object)
    props_dilated = PIG.get_region_props(mask_object_dil)
    coords_centered = props["coords_centered"]

    label_dim_max_dilated = props_dilated["label_dim_max"]
    coords_centered_dilated = props_dilated["coords_centered"]

    value = [
        randint(max(label_dim_max_dilated[0], min_z),
                min(mask.shape[0] - label_dim_max_dilated[0] - 1, max_z)),
        randint(max(label_dim_max_dilated[1], min_y),
                min(mask.shape[1] - label_dim_max_dilated[1] - 1, max_y)),
        randint(max(label_dim_max_dilated[2], min_x),
                min(mask.shape[2] - label_dim_max_dilated[2] - 1, max_x))
        ]

    coords = np.add(coords_centered, value)
    coords_dilated = np.add(coords_centered_dilated, value)

    # calculate overlap to existing objects
    # in case of multiple lumen we need an additional mask with lumen as foreground
    overlap = np.count_nonzero(mask_lumen[coords_dilated[:, 0], coords_dilated[:, 1], coords_dilated[:, 2]])
    overlap_with_mask = np.count_nonzero(mask[coords_dilated[:, 0], coords_dilated[:, 1], coords_dilated[:, 2]])

    if overlap / props_dilated["area"] <= max_overlap and overlap_with_mask >= props_dilated["area"]:
        mask_label[coords[:, 0], coords[:, 1], coords[:, 2]] = 0 # lumen_num
        mask_lumen[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        lumen_num += 1

    if lumen_num-2 >= max_lumen_num:
        break

mask_label[mask_label>0] = 255
tiff.imwrite(path_output, mask_label)