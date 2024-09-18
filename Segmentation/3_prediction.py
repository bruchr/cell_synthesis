# **In case of problems or questions, please first check the list of [Frequently Asked Questions (FAQ)](https://stardist.net/docs/faq.html).**
# 
# Please shutdown all other training/prediction notebooks before running this notebook (as those might occupy the GPU memory otherwise).

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.measure import regionprops

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap
from stardist.models import StarDist3D

from pathlib import Path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Inference stardist")
    parser.add_argument("data_folder", type=str)
    parser.add_argument("base_dir", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("--plots", type=bool, default=False)
    args = parser.parse_args()

    np.random.seed(6)
    lbl_cmap = random_label_cmap()

    # # Data
    # 
    # We assume that data has already been downloaded in via notebook [1_data.ipynb](1_data.ipynb).  
    # We now load images from the sub-folder `test` that have not been used during training.

    data_folder = Path(vars(args)['data_folder'])
    base_dir = vars(args)['base_dir']
    model_name = vars(args)['model_name']
    plots = vars(args)['plots']
    # data_folder = Path('C:/Users/xx3662/Desktop/Projekte_Daten/Segmentation/Test/Syn_MultiClass_Test/test_data_v1_split0d05/images')
    # base_dir = 'C:/Users/xx3662/Desktop/Projekte_Daten/Segmentation/Models_Runs/Stardist/Syn_MultiClass'
    # model_name = 'td_v1_split0d5_run1_20230724'

    def dict_to_labelimg(label, c_dict):
        c_label = np.zeros_like(label)
        props = regionprops(label)
        for prop in props:
            c_label[tuple(prop.coords.T)] = c_dict[prop.label]
        return c_label

    X_paths = sorted(data_folder.glob('*.tif'))
    for x_ in X_paths:
        print(x_)
    X = list(map(imread,X_paths))

    n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]
    axis_norm = (0,1,2)   # normalize channels independently
    # axis_norm = (0,1,2,3) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

    # show all test images
    if plots:
        fig, ax = plt.subplots(1,3, figsize=(16,16))
        for i,(a,x) in enumerate(zip(ax.flat, X)):
            a.imshow(x[x.shape[0]//2],cmap='gray')
            a.set_title(i)
        [a.axis('off') for a in ax.flat]
        plt.tight_layout()

    # # Load trained model
    # 
    # If you trained your own StarDist model (and optimized its thresholds) via notebook [2_training.ipynb](2_training.ipynb), then please set `demo_model = False` below.
    demo_model = False

    if demo_model:
        print (
            "NOTE: This is loading a previously trained demo model!\n"
            "      Please set the variable 'demo_model = False' to load your own trained model.",
            file=sys.stderr, flush=True
        )
        model = StarDist3D.from_pretrained('3D_demo')
    else:
        model = StarDist3D(None, name=model_name, basedir=base_dir)


    # ## Prediction
    # 
    # Make sure to normalize the input image beforehand or supply a `normalizer` to the prediction function.
    # 
    # Calling `model.predict_instances` will
    # - predict object probabilities and star-convex polygon distances (see `model.predict` if you want those)
    # - perform non-maximum suppression (with overlap threshold `nms_thresh`) for polygons above object probability threshold `prob_thresh`.
    # - render all remaining polygon instances in a label image
    # - return the label instances image and also the details (coordinates, etc.) of all remaining polygons

    for i in range(len(X)):
        img = normalize(X[i], 1, 99.8, axis=axis_norm)
        labels, details = model.predict_instances(img, n_tiles=model._guess_n_tiles(img), show_tile_progress=False)
        output_path = X_paths[i].parent / ('results_' + model_name) / ('seg_' + X_paths[i].name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_tiff_imagej_compatible(output_path, labels, axes='ZYX')
        if 'class_id' in details:
            cls_dict = dict((i+1,c) for i,c in enumerate(details['class_id']))
            c_labels = dict_to_labelimg(labels, cls_dict)
            output_path_clabels = X_paths[i].parent / ('results_' + model_name) / ('class_' + X_paths[i].name)
            save_tiff_imagej_compatible(output_path_clabels, c_labels, axes='ZYX')

    # plt.figure(figsize=(13,10))
    # z = max(0, img.shape[0] // 2 - 5)
    # plt.subplot(121)
    # plt.imshow((img if img.ndim==3 else img[...,:3])[z], clim=(0,1), cmap='gray')
    # plt.title('Raw image (XY slice)')
    # plt.axis('off')
    # plt.subplot(122)
    # plt.imshow((img if img.ndim==3 else img[...,:3])[z], clim=(0,1), cmap='gray')
    # plt.imshow(labels[z], cmap=lbl_cmap, alpha=0.5)
    # plt.title('Image and predicted labels (XY slice)')
    # plt.axis('off')

    # ## Save predictions
    # 
    # Uncomment the lines in the following cell if you want to save the example image and the predicted label image to disk.


    # save_tiff_imagej_compatible('example_image.tif', img, axes='ZYX')
    # save_tiff_imagej_compatible('example_labels.tif', labels, axes='ZYX')


    # # Example results


    def example(model, i, show_dist=True):
        img = normalize(X[i], 1,99.8, axis=axis_norm)
        labels, details = model.predict_instances(img, n_tiles=model._guess_n_tiles(img))

        plt.figure(figsize=(13,8))
        z = img.shape[0] // 2
        y = img.shape[1] // 2
        img_show = img if img.ndim==3 else img[...,:3]    
        plt.subplot(221); plt.imshow(img_show[z],   cmap='gray', clim=(0,1)); plt.axis('off'); plt.title('XY slice')
        plt.subplot(222); plt.imshow(img_show[:,y], cmap='gray', clim=(0,1)); plt.axis('off'); plt.title('XZ slice')
        plt.subplot(223); plt.imshow(img_show[z],   cmap='gray', clim=(0,1)); plt.axis('off'); plt.title('XY slice')
        plt.imshow(labels[z], cmap=lbl_cmap, alpha=0.5)
        plt.subplot(224); plt.imshow(img_show[:,y], cmap='gray', clim=(0,1)); plt.axis('off'); plt.title('XZ slice')
        plt.imshow(labels[:,y], cmap=lbl_cmap, alpha=0.5)
        plt.tight_layout()
        plt.show()


    # example(model, 0)


    # example(model, 1)


    # example(model, 2)


