import random

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import mytransforms
from utils.utils import paths_images_in_folder
from skimage.io import imread


class CycleGANDataset(Dataset):
    """Standard CycleGAN dataset."""

    def __init__(self, opt):
        """
        Args:
            A_folder (string): Path to images from domain A
            B_folder (string): Path to images from domain B
            transforms (callable, optional): Optional transforms to be applied on sample.
        """
        self.opt = opt
        self.img_dim = opt["img_dim"]
        self.folder = opt["dataset_folder"]
        self.train = self.opt["mode"] == "train"
        self.A_paths = paths_images_in_folder(self.folder, self.opt["mode"], "A")
        self.A_size = len(self.A_paths)
        if not self.train or opt["virtual_dataset_multiplicator"] == 1 or opt["virtual_dataset_multiplicator"] is None:
            self.A_size_virt = len(self.A_paths)
        else:
            self.A_size_virt = np.round(len(self.A_paths) * opt["virtual_dataset_multiplicator"])
        self.A_transforms = get_transforms(self.opt)
        if self.train:
            self.B_paths = [path.replace(self.opt["mode"]+"A", self.opt["mode"]+"B") for path in self.A_paths]
            self.Seg_transforms = get_transforms(self.opt)
            self.Seg_paths = paths_images_in_folder(self.folder, self.opt["mode"], "Seg")


    def __len__(self):
        """Returns the number of images in domain A
        """
        return self.A_size_virt

    def __getitem__(self, index):
        """ Returns an image of each domain A and B.
        For training, every image in domain A is taken, while B is randomly shuffled.
        """
        A_path = self.A_paths[index % self.A_size]  # assure the index is in range of A
        A_img = imread(A_path)
        if self.train:
            B_path = self.B_paths[index % self.A_size]
            B_img = imread(B_path)
            Seg_path = self.Seg_paths[index % self.A_size]
            Seg_img = imread(Seg_path)
            samples = self.A_transforms({'imageA': A_img, 'imageB': B_img})
            samples_Seg = self.Seg_transforms({'imageA': Seg_img, 'imageB': None})
            return {'A': samples['imageA'], 'B': samples['imageB'], 'A_path': A_path, 'B_path': B_path, 'Seg': samples_Seg['imageA']}
        else:
            samples = self.A_transforms({'imageA': A_img, 'imageB': None})
            return {'A': samples['imageA'], 'A_path': A_path}           


def get_transforms(opt):
    transforms_list = []
    img_dim = opt["img_dim"]
    crop_size = opt["crop_size"]

    transforms_list.append(mytransforms.Channel_Order(img_dim, crop_size))
    
    if opt["mode"] == "train":
        if opt["preprocess"] == "crop":
            transforms_list.append(mytransforms.RandomCrop(crop_size, img_dim)) # Random crop
        transforms_list.append(mytransforms.RandomFlip(img_dim))  # Random horizontal flip
    # transforms_list.append(transforms.Lambda(lambda img: __changetonumpy(img)))
    transforms_list.append(mytransforms.Normalize())
    transforms_list.append(mytransforms.ToTensor())
    return transforms.Compose(transforms_list)


def __changetonumpy(img):
    img = np.asarray(img)
    img = img.astype('float32')
    return img


def __normalize(img):
    """ Normalize uint8"""
    img = 2*img/np.iinfo("uint8").max-1
    return img