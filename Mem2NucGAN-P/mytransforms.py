import random

import numpy as np
import torch

class Channel_Order(object):
    """Arrange the channels of input image"""
    def __init__(self, ndim, crop_size):
        self.ndim = ndim
        self.ndim_crop = len(crop_size) if crop_size is not None else None
    
    def __order_channels(self, image):
        if self.ndim == 2: # 2D or 3D crop
            if image.ndim == 2: # 2Dsc
                image = image[None, ...]
            elif image.ndim == 4: # 3D mc crop
                image = np.moveaxis(image, 1, 0)
            elif self.ndim_crop == 3: # 3D sc crop
                image = image[None, ...]
            elif self.ndim_crop is None or self.ndim_crop == 2: # 2D mc
                image = np.moveaxis(image, -1, 0)
            else:
                raise ValueError('Image shape / crop size does not fit with specified image dim!')
        else:
            if image.ndim == 3: # 3D sc
                image = image[None, ...]
            else: # 3D mc
                image = np.moveaxis(image, 1, 0)
        
        return image

    def __call__(self, sample):
        imageA = sample['imageA']
        imageB = sample['imageB']
        
        imageA = self.__order_channels(imageA)
        if imageB is not None:
            imageB = self.__order_channels(imageB)

        # output shape
        # 3D: C x D x H x W
        # 2D: C x H x W
        
        return {'imageA': imageA, 'imageB': imageB}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        imageA = sample['imageA']
        imageB = sample['imageB']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))

        if imageB is not None:
            return {'imageA': torch.from_numpy(imageA), 'imageB': torch.from_numpy(imageB)}
        else:
            return {'imageA': torch.from_numpy(imageA), 'imageB': None}



class Normalize(object):
    """ Normalize uint8"""
    def __call__(self, sample):
        imageA = sample['imageA']
        
        if imageA.dtype == 'uint8' or imageA.dtype == 'uint16':
            dtype = imageA.dtype
            imageA = imageA.astype(np.float32)
            imageA = 2*imageA/np.iinfo(dtype).max - 1
        else:
            imageA = imageA.astype(np.float32)

        
        imageB = sample['imageB']
        if imageB is not None:
            if imageB.dtype == 'uint8' or imageB.dtype == 'uint16':
                dtype = imageB.dtype
                imageB = imageB.astype(np.float32)
                imageB = 2*imageB/np.iinfo(dtype).max - 1
            else:
                imageB = imageB.astype(np.float32)
    
        return {'imageA': imageA, 'imageB': imageB}



class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
        img_dim (int): Dimensionality of the images (2D or 3D)
    """

    def __init__(self, output_size, img_dim):
        self.img_dim = img_dim
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size) if img_dim==2 else (output_size, output_size, output_size)
        else:
            if len(output_size) == 3 and img_dim == 2:
                # 3D image will be cropped to 2D
                assert any(el == 1 for el in output_size), 'If ndim is 2 and length of output size is 3D, one element needs to be one!'
            else:
                assert len(output_size) == img_dim
            self.output_size = output_size

    def __call__(self, sample):
        imageA = sample['imageA']
        imageB = sample['imageB']

        if self.img_dim ==2 and len(self.output_size) == 2:
            h, w = imageA.shape[-2:]
            new_h, new_w = self.output_size

            top = np.random.randint(0, h - new_h) if new_h!=h else 0
            left = np.random.randint(0, w - new_w) if new_w!=w else 0

            imageA = imageA[...,
                          top: top + new_h,
                          left: left + new_w]
            if imageB is not None:
                imageB = imageB[...,
                            top: top + new_h,
                            left: left + new_w]
        
        else:
            d, h, w = imageA.shape[-3:]
            new_d, new_h, new_w = self.output_size

            upper = np.random.randint(0, d - new_d) if new_d!=d else 0
            top = np.random.randint(0, h - new_h) if new_h!=h else 0
            left = np.random.randint(0, w - new_w) if new_w!=w else 0

            imageA = imageA[...,
                            upper: upper + new_d,
                            top: top + new_h,
                            left: left + new_w]
            if imageB is not None:
                imageB = imageB[...,
                                upper: upper + new_d,
                                top: top + new_h,
                                left: left + new_w]
            squeeze = np.nonzero(np.asarray(imageA.shape)[1:] == 1)[0]+1
            imageA = np.squeeze(imageA, axis=tuple(squeeze))
            if imageB is not None:
                squeeze = np.nonzero(np.asarray(imageB.shape)[1:] == 1)[0]+1
                imageB = np.squeeze(imageB, axis=tuple(squeeze))

        return {'imageA': imageA, 'imageB': imageB}


class RandomFlip(object):
    """Flip or rotate (90°) image and label image (label-changing transformation)."""

    def __init__(self, img_dim, p=0.5):
        """
        :param ndim: Dimensionality of the images (2D or 3D)
            :type p: int
        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.ndim = img_dim
        self.p = p

    def __call__(self, sample):
        """
         :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """
        imgA = sample['imageA']
        imgB = sample['imageB']

        # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())

        if random.random() < self.p:

            # img.shape: (Height, Width)
            h = random.randint(0, 2)
            if h == 0:
                # Flip left-right
                imgA = np.flip(imgA, axis=1) if self.ndim ==2 else np.flip(imgA, axis=2)
                if imgB is not None:
                    imgB = np.flip(imgB, axis=1) if self.ndim ==2 else np.flip(imgB, axis=2)
            elif h == 1:
                # Flip up-down
                imgA = np.flip(imgA, axis=2) if self.ndim ==2 else np.flip(imgA, axis=3)
                if imgB is not None:
                    imgB = np.flip(imgB, axis=2) if self.ndim ==2 else np.flip(imgB, axis=3)
            elif h == 2:
                # Rotate 90°
                imgA = np.rot90(imgA, axes=(1, 2)) if self.ndim ==2 else np.rot90(imgA, axes=(2, 3))
                if imgB is not None:
                    imgB = np.rot90(imgB, axes=(1, 2)) if self.ndim ==2 else np.rot90(imgB, axes=(2, 3))

        return {'imageA': imgA, 'imageB': imgB}