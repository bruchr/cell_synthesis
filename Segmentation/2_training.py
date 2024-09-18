# **In case of problems or questions, please first check the list of [Frequently Asked Questions (FAQ)](https://stardist.net/docs/faq.html).**
# 
# Please shutdown all other training/prediction notebooks before running this notebook (as those might occupy the GPU memory otherwise).

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as plot_cm

from glob import glob
from skimage.measure import regionprops
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D
from stardist.utils import mask_to_categorical
from stardist.plot import render_label

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Training stardist")
    parser.add_argument("data_folder", type=str)
    parser.add_argument("base_dir", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--plots", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_patch_size", type=int, nargs='+', default=[96,256,256])
    args = parser.parse_args()
    
    # np.random.seed(42)
    lbl_cmap = random_label_cmap()

    # # Data
    # Training data (for input `X` with associated label masks `Y`) can be provided via lists of numpy arrays, where each image can have a different size. Alternatively, a single numpy array can also be used if all images have the same size.  
    # Input images can either be three-dimensional (single-channel) or four-dimensional (multi-channel) arrays, where the channel axis comes last. Label images need to be integer-valued.
    
    data_folder = vars(args)['data_folder']
    base_dir = vars(args)['base_dir']
    model_name = vars(args)['model_name']
    n_classes = vars(args)['n_classes']
    plots = vars(args)['plots']
    epochs = vars(args)['epochs']
    train_patch_size = tuple(vars(args)['train_patch_size'])

    if len(train_patch_size) != 3:
        raise ValueError(f'train_patch_size needs to be 3 values (z,y,x)! Recieved: {train_patch_size}')
    

    # data_folder = 'C:/Users/xx3662/Desktop/Projekte_Daten/Segmentation/Training_Data/Stardist/TD_3D_Mario_KP4_R6_SingleClass_V1'
    # base_dir = 'C:/Users/xx3662/Desktop/Projekte_Daten/Segmentation/Models_Runs/Stardist'
    # model_name = 'td_3D_Mario_KP4_R6_SingleClass_V1_rV1_20230710'
    # n_classes = None
    # plots = False

    def convert2dict(label:np.ndarray, clabel_path:str) -> dict:
        clabel = imread(clabel_path)
        props = regionprops(label)
        get_class = lambda prop: clabel[tuple(prop.coords[0,:].T)]
        c_dict = dict((prop.label, get_class(prop)) for prop in props)
        return c_dict

    X = sorted(glob(data_folder + '/train/images/*.tif'))
    Y = sorted(glob(data_folder + '/train/masks/*.tif'))
    if n_classes is not None:
        C = sorted(glob(data_folder + '/train/cmasks/*.tif'))
        assert (Path(x).name==Path(c).name for x,c in zip(X,C))
    assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))
    print(f'Number of training images and masks: {len(X)}, {len(Y)}')

    X = list(map(imread,X))
    Y = list(map(imread,Y))
    if n_classes is not None:
        C = list(map(convert2dict,Y,C))
    n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]

    def plot_img_label(img, lbl, cls_dict, n_classes=2, img_title="image", lbl_title="label", cls_title="classes", **kwargs):
        center_slice = img.shape[0]//2
        img = img[center_slice,...]
        lbl = lbl[center_slice,...]
        c = mask_to_categorical(lbl, n_classes=n_classes, classes=cls_dict)
        res = np.zeros(lbl.shape, np.uint16)
        for i in range(1,c.shape[-1]):
            m = c[...,i]>0
            res[m] = i
        class_img = plot_cm.tab20(res)
        class_img[...,:3][res==0] = 0 
        class_img[...,-1][res==0] = 1
        
        fig, (ai,al,ac) = plt.subplots(1,3, figsize=(17,7), gridspec_kw=dict(width_ratios=(1.,1,1)))
        im = ai.imshow(img, cmap='gray')
        #fig.colorbar(im, ax = ai)
        ai.set_title(img_title)    
        al.imshow(render_label(lbl, .8*normalize(img, clip=True), normalize_img=False, alpha_boundary=.8,cmap=lbl_cmap))
        al.set_title(lbl_title)
        ac.imshow(class_img)
        ac.imshow(render_label(res, .8*normalize(img, clip=True), normalize_img=False, alpha_boundary=.8, cmap=plot_cm.tab20))
        ac.set_title(cls_title)
        plt.tight_layout()    
        for a in ai,al,ac:
            a.axis("off")
        return ai,al,ac
    if plots:
        ax = plot_img_label(X[0],Y[0],C[0], n_classes=n_classes)
        plt.show()

    # Normalize images and fill small label holes.
    axis_norm = (0,1,2)   # normalize channels independently
    # axis_norm = (0,1,2,3) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 3 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X, desc='Image Normalization')]
    Y = [fill_label_holes(y) for y in tqdm(Y, desc='Fill Label Holes')]


    # Split into train and validation datasets.
    assert len(X) > 1, "not enough training data"
    # rng = np.random.RandomState(42)
    rng = np.random.RandomState()
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    if n_classes is not None:
        C_val = [C[i] for i in ind_val]
        C_trn = [C[i] for i in ind_train]
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))



    # # Configuration
    # print(Config3D.__doc__)

    extents = calculate_extents(Y)
    anisotropy = tuple(np.max(extents) / extents)
    print('empirical anisotropy of labeled objects = %s' % str(anisotropy))

    # 96 is a good default choice (see 1_data.ipynb)
    n_rays = 96

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = True and gputools_available()

    if not use_gpu:
        print(f'#-#-#-#-#-#-#-#- GPU is NOT used! -#-#-#-#-#-#-#-#')

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

    # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    conf = Config3D (
        rays             = rays,
        grid             = grid,
        anisotropy       = anisotropy,
        use_gpu          = use_gpu,
        n_channel_in     = n_channel,
        n_classes        = n_classes,
        # adjust for your data below (make patch size as large as possible)
        train_patch_size = train_patch_size, # (96,256,256), # (48,96,96),
        train_batch_size = 2,
        train_epochs = epochs,
    )
    print(conf)
    vars(conf)

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8, total_memory=47000)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)


    # **Note:** The trained `StarDist3D` model will *not* predict completed shapes for partially visible objects at the image boundary.


    model = StarDist3D(conf, name=model_name, basedir=base_dir)
    # model = StarDist3D(None, name='td_mario_ht29_V3_run1_20231121 - Kopie', basedir='C:/Users/xx3662/Desktop/Projekte_Daten/Segmentation/Models_Runs/Stardist')

    # Check if the neural network has a large enough field of view to see up to the boundary of most objects.
    median_size = calculate_extents(Y, np.median)
    fov = np.array(model._axes_tile_overlap('ZYX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    # # Data Augmentation
    # You can define a function/callable that applies augmentation to each batch of the data generator.  
    # We here use an `augmenter` that applies random rotations, flips, and intensity changes, which are typically sensible for (3D) microscopy images (but you can disable augmentation by setting `augmenter = None`).
    def random_fliprot(img, mask, axis=None): 
        if axis is None:
            axis = tuple(range(mask.ndim))
        axis = tuple(axis)
                
        assert img.ndim>=mask.ndim
        perm = tuple(np.random.permutation(axis))
        transpose_axis = np.arange(mask.ndim)
        for a, p in zip(axis, perm):
            transpose_axis[a] = p
        transpose_axis = tuple(transpose_axis)
        img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim))) 
        mask = mask.transpose(transpose_axis) 
        for ax in axis: 
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=ax)
                mask = np.flip(mask, axis=ax)
        return img, mask 

    def random_intensity_change(img):
        img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
        return img

    def augmenter(x, y):
        """Augmentation of a single input/label image pair.
        x is an input image
        y is the corresponding ground-truth label image
        """
        # Note that we only use fliprots along axis=(1,2), i.e. the yx axis 
        # as 3D microscopy acquisitions are usually not axially symmetric
        x, y = random_fliprot(x, y, axis=(1,2))
        x = random_intensity_change(x)
        return x, y



    # # Training

    # We recommend to monitor the progress during training with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard). You can start it in the shell from the current working directory like this:
    # 
    #     $ tensorboard --logdir=.
    # 
    # Then connect to [http://localhost:6006/](http://localhost:6006/) with your browser.
    # 
    if n_classes is not None:
        model.train(X_trn, Y_trn, classes=C_trn, validation_data=(X_val,Y_val,C_val), augmenter=augmenter)
    else:
        model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter)


    # # Threshold optimization
    # While the default values for the probability and non-maximum suppression thresholds already yield good results in many cases, we still recommend to adapt the thresholds to your data. The optimized threshold values are saved to disk and will be automatically loaded with the model.

    model.optimize_thresholds(X_val, Y_val)







    # # Evaluation and Detection Performance
    # Besides the losses and metrics during training, we can also quantitatively evaluate the actual detection/segmentation performance on the validation data by considering objects in the ground truth to be correctly matched if there are predicted objects with overlap (here [intersection over union (IoU)](https://en.wikipedia.org/wiki/Jaccard_index)) beyond a chosen IoU threshold $\tau$.
    # 
    # The corresponding matching statistics (average overlap, accuracy, recall, precision, etc.) are typically of greater practical relevance than the losses/metrics computed during training (but harder to formulate as a loss function). 
    # The value of $\tau$ can be between 0 (even slightly overlapping objects count as correctly predicted) and 1 (only pixel-perfectly overlapping objects count) and which $\tau$ to use depends on the needed segmentation precision/application.
    # 
    # Please see `help(matching)` for definitions of the abbreviations used in the evaluation below and see the Wikipedia page on [Sensitivity and specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) for further details.

    # help(matching)

    # First predict the labels for all validation images:
    Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
                for x in tqdm(X_val, desc='Predict Instances')]

    # Plot a GT/prediction example  
    # plot_img_label(X_val[0],Y_val[0], lbl_title="label GT (XY slice)")
    # plot_img_label(X_val[0],Y_val_pred[0], lbl_title="label Pred (XY slice)")


    # Choose several IoU thresholds $\tau$ that might be of interest and for each compute matching statistics for the validation data.
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus, desc='Compute Matching Statistics')]


    # Example: Print all available matching statistics for $\tau=0.7$

    if plots:
        stats[taus.index(0.7)]
        # Plot the matching statistics and the number of true/false positives/negatives as a function of the IoU threshold $\tau$. 
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

        for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
            ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax1.set_xlabel(r'IoU threshold $\tau$')
        ax1.set_ylabel('Metric value')
        ax1.grid()
        ax1.legend()

        for m in ('fp', 'tp', 'fn'):
            ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax2.set_xlabel(r'IoU threshold $\tau$')
        ax2.set_ylabel('Number #')
        ax2.grid()
        ax2.legend()
        plt.show()