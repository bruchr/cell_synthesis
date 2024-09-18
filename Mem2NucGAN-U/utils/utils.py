import os
from pathlib import Path

import numpy as np
import tifffile as tiff
import torchvision.transforms as transforms


"""Simple helper functions"""


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.TIFF', '.tif', '.TIF']

def num_images_in_folder(folder, include_subfolders=True):
    num_images = 0
    if include_subfolders:
        for (_, _,filenames) in os.walk(folder):
            for filename in filenames:
                if is_image_file(filename):
                    num_images = num_images+1
    else:
        print("Has to be implemented")
    return num_images


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def paths_images_in_folder(folder, mode, domain,include_subfolders=True):
    image_paths = []
    folder = os.path.join(folder, mode + domain)
    if include_subfolders:
        for (root, _,filenames) in os.walk(folder):
            for filename in filenames:
                if is_image_file(filename):
                    image_paths.append(os.path.join(root, filename))
    if len(image_paths) == 0:
        raise ValueError(f'No images found in folder: {folder}')
    return image_paths


def save_image_batch(batch, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(batch['A'].size()[0]):
        transform = transforms.ToPILImage()
        A_img = transform(batch['A'][i])
        A_img.save(os.path.join(folder, f"A_{i}.png"))
        B_img = transform(batch['B'][i])
        B_img.save(os.path.join(folder, f"B_{i}.png"))


def inference_save_images(batch, folder, filenames, img_dim):
    if not os.path.exists(folder):
        os.makedirs(folder)    
    for i in range(batch.shape[0]):
        if img_dim == 3:
            batch = np.moveaxis(batch, 1, 2)
        else:
            batch = np.moveaxis(batch, 1, -1)
        im = batch[i,...]
        filename = Path(filenames[i]).name
        path = Path(folder).joinpath('transformed_' + filename)
        tiff.imsave(path, im, imagej=True if img_dim==3 else False)


def tensor2np(input_image, imtype):
    """ Converts a Tensor image to a numpy image

    Arguments:
        input_image {tensor} -- input image tensor
        imtype {string} -- Data type of the numpy image (e.g. uint8, uint16)

    Returns:
        [np.array] -- image as numpy array
    """
    image_tensor = input_image.data
    image_numpy = image_tensor.cpu().float().numpy()
    # (image+1)/2 because tanh (GAN otuput) is [-1,1] therefore +1 => [0,2] and /2 => [0,1] then * max of image value stretches to imagesize
    image_numpy = (image_numpy + 1) / 2.0 * np.iinfo(imtype).max  # 255.0 np.iinfo(imtype).max  
    return image_numpy.astype(imtype)
