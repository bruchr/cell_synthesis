import os.path
import time
from typing import Tuple, Optional, List, Dict, Any

import torch
import numpy as np
import scipy.ndimage
import tifffile as tiff
from skimage.transform import rescale
from torch.nn.functional import conv3d

from utils import load_data

class OpticalSystemSimulation:
    path_output: str
    paths_psf: List[str]
    path_spheroid_mask: str
    z_append: int
    shape_des: np.ndarray
    brightness_red_fuction: str
    brightness_red_factor: int
    gpu_conv: int


    def __init__(self) -> None:
        pass

    def import_settings(self, params: Dict[str, Any]) -> None:
        """
        Sets the given parameters.

        Args:
            params: A dict containing all parameters for the simulation process

        Returns:
            None
        """
        self.path_output = params["path_output"]
        self.paths_psf = params["paths_psf"]
        self.path_spheroid_mask = params["path_spheroid_mask"]
        self.z_append = params["z_append"]
        self.shape_des = params["shape_des"]
        self.brightness_red_fuction = params["brightness_red_fuction"]
        self.brightness_red_factor = params["brightness_red_factor"]
        self.gpu_conv = params["gpu_conv"]

    @staticmethod
    def load_images(image_files: List[str]) -> List[np.ndarray]:
        images = list()
        for image_file in image_files:
            images.append(tiff.imread(image_file))

        return images

    def brightness_reduction(self, image_sim: np.ndarray, function: str = "f3", factor: float = 200,
                             save_mask_to: Optional[str] = None) -> np.ndarray:
        """
        Reduces the brightness of pixels inside the cell culture.

        Args:
            image_sim: phantom image
            minimum: the minimum percentage of intensity that no pixel should go below
            save_mask_to: optional path, to save the mask to

        Returns:
            The image with reduced intensity inside the cell culture
        """
        z, y, x = image_sim.shape

        true_z = np.copy(image_sim.shape[0]); true_z -= self.z_append # True shape without appended slices
        scale_z = true_z / self.shape_des[0]

        spheroid_mask = np.zeros((z,y,x), dtype=np.float32)
        spheroid_mask[1:,...] = load_data(self.path_spheroid_mask)[:-1,...]

        if function=="fOld":
            # b_funct = lambda i: np.maximum(np.minimum(((i / z + 1) ** (-6) + 0.2), 1), 0) # fOld
            b_funct = lambda i: np.maximum(np.minimum( np.add(np.power(np.add(np.divide(i,z,out=i),1,out=i),-6,out=i),0.2,out=i) , 1, out=i), 0, out=i)
        elif function=="f1":
            # b_funct = lambda i: np.maximum(np.minimum(((i + 5) / z + 1) ** (-6) , 1), 0) # f1
            b_funct = lambda i: np.maximum(np.minimum( np.power(np.add(np.divide(np.add(i,5,out=i),z,out=i),1,out=i),-6,out=i) , 1, out=i), 0, out=i)
        elif function=="f2p" or function=="f2":
            if function=="f2": factor = 150
            # b_funct = lambda i: np.maximum(np.minimum((i / factor + 1) ** (-6) , 1), 0) # f2 with parameter
            b_funct = lambda i: np.maximum(np.minimum( np.power(np.add(np.divide(i,factor,out=i),1,out=i),-6,out=i) , 1, out=i), 0, out=i)
        elif function=="f3p" or function=="f3":
            if function=="f3": factor = 200
            # b_funct = lambda i: np.maximum(np.minimum(((i-10) / factor + 1) ** (-6) , 1), 0) # f3 with parameter
            b_funct = lambda i: np.maximum(np.minimum( np.power(np.add(np.divide(np.subtract(i,10,out=i),factor,out=i),1,out=i),-6,out=i) , 1, out=i), 0, out=i)
        elif function=="None":
            return image_sim # No Reduction
        
        spheroid_mask[0,...] = 0
        for z_i in range(1, spheroid_mask.shape[0]):
            spheroid_mask[z_i, ...] += spheroid_mask[z_i-1, ...]
        spheroid_mask /= scale_z
        spheroid_mask = b_funct(spheroid_mask)

        if save_mask_to is not None:
            tiff.imwrite(save_mask_to, (spheroid_mask * 255).astype(np.uint8))

        image_sim *= spheroid_mask
        return image_sim

    @staticmethod
    def convolve_image_multiple_psf(image: np.ndarray, psfs: List[np.ndarray], gpu_conv: bool = True) -> np.ndarray:
        num_of_psf = len(psfs)

        if num_of_psf == 1:
            return OpticalSystemSimulation.convolve_image(image, psfs[0], gpu_conv=gpu_conv)

        images = list()
        z, y, x = image.shape
        num_of_psf = len(psfs)
        num_of_slices = (num_of_psf * 2 - 1)
        len_of_slices = int(z / num_of_slices)

        for n, psf in enumerate(psfs):
            if n == 0:
                offset = 0
                width = 2 * len_of_slices
            elif n == num_of_psf - 1:
                offset = (n * 2 - 1) * len_of_slices
                width = 2 * len_of_slices + (z - num_of_slices * len_of_slices)
            else:
                offset = (n * 2 - 1) * len_of_slices
                width = 3 * len_of_slices

            curr_image = np.zeros_like(image)
            curr_image[offset:offset + width] = OpticalSystemSimulation.convolve_image(image[offset: offset + width],
                                                                                       psf, gpu_conv=gpu_conv)
            images.append(curr_image)

        image_final = images[-1]

        for n in range(num_of_psf * 2 - 1):
            if n % 2 == 0:
                image_final[n * len_of_slices: n * len_of_slices + len_of_slices] = \
                    images[int(n / 2)][n * len_of_slices: n * len_of_slices + len_of_slices]
            else:
                # transition between two point spread functions
                offset = n * len_of_slices
                for i in range(len_of_slices):
                    image_final[offset + i] = (1 - i / len_of_slices) * images[int(n / 2)][offset + i] \
                                              + (i / len_of_slices) * images[int(n / 2) + 1][offset + i]

        return image_final

    @staticmethod
    def convolve_image(img: np.ndarray, psf: np.ndarray,  gpu_conv: bool = True) -> np.ndarray:
        """
        Convolves the given image with the given point spread function.

        Args:
            img: image to be convolved with the given psf
            psf: point spread function to be used for the convolution
            gpu_conv: if convolution should be performed on GPU

        Returns:
            The convolved image
        """
        psf = psf.astype(np.float32)

        # Check if cuda is available to run convolution on gpu
        if gpu_conv and torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True

            pad = tuple(np.ceil((np.asarray(psf.shape) - 1)/2).astype(int))
            # Unsymetric padding required for even psf shape, but not supported in torch.
            # Use larger one and remove unwanted part of the image later.
            even = np.logical_not(np.asarray(psf.shape)%2).astype(int)

            img_c = torch.from_numpy(img[None, ...]).to(device)
            psf_c = torch.from_numpy(psf[None, None, ...]).to(device)
            img = conv3d(img_c, psf_c, padding=pad)[0, even[0]:, even[1]:, even[2]:].cpu().numpy() # padding_mode=zeros

        else:
            scipy.ndimage.convolve(img, np.flip(psf), mode="constant", cval=0, output=img)

        return img

    def run(self, image: np.ndarray, mask_num: int, img_num: int, verbose: bool = False,
            save_interim_images: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates the optical system for the given image

        Args:
            image: image to process
            mask_num: number of the current mask used for placement of nuclei (is used for saving the images when save_interim_images is set)
            img_num: number of the current image (is used for saving the images when save_interim_images is set)
            verbose: if set, the command line output is more verbose. If verbose==2, required times for substeps are displayed.
            save_interim_images: if set, all interim results are saved

        Returns:
            The processed image
        """
        if verbose:
            s_OSS = time.time()
            print("\n--#--#--#-- Optical System Simulation --#--#--#--")
            print("Calculating and applying brightness reduction ...")
            s_OSS_bred = time.time()


        ### Brightness Reduction
        reduction_mask_out = None
        if save_interim_images:
            reduction_mask_out = os.path.join(self.path_output, "{:03d}_{:03d}_brightness_red_mask.tif".format(mask_num, img_num))

        image = self.brightness_reduction(image, save_mask_to=reduction_mask_out, function=self.brightness_red_fuction, factor=self.brightness_red_factor)

        if verbose == 2:
            d_OSS_bred = time.time() - s_OSS_bred; print(f"Calculating and applying brightness reduction required {d_OSS_bred:.1f} s -> {d_OSS_bred/60:.1f} min")
        if save_interim_images:
            tiff.imwrite(os.path.join(self.path_output, "{:03d}_{:03d}_brightness_red_z.tif".format(mask_num, img_num)), image)


        ### PSF Simulation
        if len(self.paths_psf) != 0:
            if verbose:
                print("Simulation of PSF effect ...")
                if self.gpu_conv and torch.cuda.is_available():
                    print("GPU is used for convolution")
                else:
                    print("CPU is used for convolution")
                s_OSS_conv = time.time()

            image = self.convolve_image_multiple_psf(image, self.load_images(self.paths_psf), gpu_conv=self.gpu_conv)

            if verbose == 2:
                d_OSS_conv = time.time() - s_OSS_conv; print(f"Simulation of PSF effect required {d_OSS_conv:.1f} s -> {d_OSS_conv/60:.1f} min")
            if save_interim_images:
                tiff.imwrite(os.path.join(self.path_output, "{:03d}_{:03d}_convolved.tif".format(mask_num, img_num)),
                            image)
        else:
            print("Skipping PSF convolution")
        

        if verbose:
            d_OSS = time.time() - s_OSS
            print(f"Optical System Simulation completed")
            print(f"Time required: {d_OSS:.1f} s -> {d_OSS/60:.1f} min")

        return image
