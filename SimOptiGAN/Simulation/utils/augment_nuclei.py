import os
import random

import numpy as np
import scipy
from skimage.measure import regionprops
import tifffile as tiff

def random_rotate(img, label, p=1):

    angle_z = (-180, 180)
    angle_y = (-20, 20)
    angle_x = (-20, 20)

    if random.random() < p:
        angle_z = random.uniform(angle_z[0], angle_z[1])
        angle_y = random.uniform(angle_y[0], angle_y[1])
        angle_x = random.uniform(angle_x[0], angle_x[1])

        img = scipy.ndimage.rotate(img, angle=angle_z, axes=(1,2))
        label = scipy.ndimage.rotate(label, angle=angle_z, axes=(1,2), order=0, prefilter=False)

        if 0:#random.random() < 0.5:
            img = scipy.ndimage.rotate(img, angle=angle_y, axes=(0,2))
            label = scipy.ndimage.rotate(label, angle=angle_y, axes=(0,2), order=0, prefilter=False)

            img = scipy.ndimage.rotate(img, angle=angle_x, axes=(0,1))
            label = scipy.ndimage.rotate(label, angle=angle_x, axes=(0,1), order=0, prefilter=False)

        label[label>0] = 1
        # crop_image
        props = regionprops(label)
        assert len(props) == 1, 'Length of props is {} not 1'.format(len(props))
        bb = props[0].bbox
        img = img[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]
        label = label[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]

    return img, label


def random_scale(img, label, p=1, min_s=0.8, max_s=1.3):

    scale_x = random.uniform(min_s, max_s)
    scale_y = random.uniform(min_s, max_s)
    scale_z = random.uniform(min_s, max_s)

    if random.random() < p:
            
        img = scipy.ndimage.zoom(img, zoom=(scale_z, scale_y, scale_x))
        label = scipy.ndimage.zoom(label, zoom=(scale_z, scale_y, scale_x), order=0, prefilter=False)

        # img = np.clip(img, -1, 1)

    return img, label


def random_blur(img, label, p=1):

    if random.random() < p:
        h = random.randint(0, 1)

        # Gaussian blur
        if h == 0:
            sigma = 2.5 * random.random()
            img = scipy.ndimage.gaussian_filter(img, sigma, mode='constant')

        # Median filter
        else:
            img = scipy.ndimage.median_filter(img, size=(3, 3, 3), mode='constant')

    return img, label



def elaTrans(img, label, p=1):
    """Elastic transform of an image: cropping or zero-padding is used to keep the image shape constant (label-changing
    transformation).
    Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
            Convolutional Neural Networks applied to Visual Document Analysis", in
            Proc. of the International Conference on Document Analysis and
            Recognition, 2003.
    https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random.random() < p:
        alpha = [50,200,200]
        sigma = [4,7.5,7.5]

        random_state = np.random.RandomState(None)

        shape = img.shape
        dz = scipy.ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma[0], mode="constant", cval=0) * alpha[0]
        dy = scipy.ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma[1], mode="constant", cval=0) * alpha[1]
        dx = scipy.ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma[2], mode="constant", cval=0) * alpha[2]

        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),indexing='ij')
        indices = np.reshape(z+dz, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        distorted_image = scipy.ndimage.map_coordinates(img, indices, order=1, mode='reflect')
        img = distorted_image.reshape(img.shape)
        distorted_label = scipy.ndimage.map_coordinates(label, indices, order=0, mode='reflect')
        label = distorted_label.reshape(img.shape)

    return img, label



def random_contrast(img, label, p=1):

    contrast_range = (0.65, 1.1)
    gamma_range = (0.5, 1.5)
    epsilon = 1e-7,

    if random.random() < p:

        # Contrast
        img_mean, img_min, img_max = img.mean(), img.min(), img.max()
        factor = np.random.uniform(contrast_range[0], contrast_range[1])
        img = (img - img_mean) * factor + img_mean

        # Gamma
        img_mean, img_std, img_min, img_max = img.mean(), img.std(), img.min(), img.max()
        gamma = np.random.uniform(gamma_range[0], gamma_range[1])
        rnge = img_max - img_min
        img = np.power(((img - img_min) / float(rnge + epsilon)), gamma) * rnge + img_min

        if random.random() < 0.5:
            img = img - img.mean() + img_mean
            img = img / (img.std() + 1e-8) * img_std

        img = np.round(np.clip(img, 0, 255)).astype(np.uint8)

    return img, label


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Augment Phantoms")
    parser.add_argument("input_path", type=str,
                        help="Input folder path")
    parser.add_argument("output_path", type=str,
                        help="Output folder path")
    parser.add_argument("n_aug", type=int,
                        help="Number of augmentations per phantom")
    parser.add_argument("--include_raw", type=bool, default=True,
                        help="If raw phantom should be included in the output")
    args = parser.parse_args()

    input_path = vars(args)["input_path"]
    output_path = vars(args)["output_path"]
    no_of_imgs = vars(args)["n_aug"]
    include_raw = vars(args)["include_raw"]

    if not os.path.isdir(input_path):
        raise AttributeError("The given input path is no valid directory")
    if not os.path.isdir(output_path):
        raise AttributeError("The given output path is no valid directory")

    p=0.8

    x_, y_, z_ = [],[],[]
    for f_name in os.listdir(input_path):
        if '_label' in f_name:
            label_path = os.path.join(input_path, f_name)
            raw_name = f_name.replace('_label.tif','')

            img = tiff.imread(label_path.replace('_label',''))
            label = tiff.imread(label_path)

            if include_raw:
                out_path_img = os.path.join(output_path, raw_name + '.tif')
                out_path_label = os.path.join(output_path, raw_name + '_label.tif')
                tiff.imwrite(out_path_img, img)
                tiff.imwrite(out_path_label, label)


            for ind in range(no_of_imgs):
                img_ = np.copy(img)
                label_ = np.copy(label)

                img_, label_ = random_rotate(img_, label_, p=p)
                img_, label_ = random_scale(img_, label_, p=1)
                img_, label_ = random_blur(img_, label_, p=p)
                img_, label_ = elaTrans(img_, label_, p=p)
                img_, label_ = random_contrast(img_, label_, p=p)

                out_path_img = os.path.join(output_path, raw_name + '{:02d}.tif'.format(ind))
                out_path_label = os.path.join(output_path, raw_name + '{:02d}_label.tif'.format(ind))
                tiff.imwrite(out_path_img, img_[:,None,...], imagej=True)
                tiff.imwrite(out_path_label, label_[:,None,...], imagej=True)