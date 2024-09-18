import time
import os.path
from typing import Tuple, Dict, Any

import numpy as np
import tifffile as tiff
from skimage.transform import rescale
from skimage.util import random_noise


class CameraAcquisitionSimulation:
    exposure_percentage: float
    px_size_img: Tuple[float, float, float]
    px_size_desired: Tuple[float, float, float]
    noise_gauss: float
    noise_gauss_absolute: float
    noise_poisson_factor: float
    baseline: int
    num_of_acquisition_iterations: int

    path_output: str

    def __init__(self):
        pass

    def import_settings(self, params: Dict[str, Any]) -> None:
        """
        Sets the given parameters.

        Args:
            params: A dict containing all parameters for the simulation process

        Returns:
            None
        """
        self.exposure_percentage = params["exposure_percentage"]
        self.path_output = params["path_output"]
        self.px_size_img = params["px_size_img"]
        self.px_size_desired = params["px_size_desired"]
        self.noise_gauss = params["noise_gauss"]
        self.noise_gauss_absolute = params["noise_gauss_absolute"]
        self.noise_poisson_factor = params["noise_poisson_factor"]
        self.baseline = params["baseline"]
        self.num_of_acquisition_iterations = params["num_of_acquisition_iterations"]

    @staticmethod
    def downsample(img: np.ndarray, px_size_img: Tuple[float, float, float],
                   px_size_desired: Tuple[float, float, float], mode: str = "image"):
        """
        Resizes the given image from one pixel size to another.

        Args:
            img: image to change
            px_size_img: pixel/voxel size of the given image
            px_size_desired: pixel/voxel size that is wanted for the given image
            mode: "image" for image data or "label" for label data

        Returns:
            The resized image
        """

        if mode == "image":
            order = 1
            aa = False
        elif mode == "label":
            order = 0
            aa = False
        else:
            raise AttributeError("Unsupported mode: {}".format(mode))

        scale = np.asarray(px_size_img) / np.asarray(px_size_desired)
        img = rescale(img, (scale[0], scale[1], scale[2]), order=order, channel_axis=None, preserve_range=True,
                      anti_aliasing=aa)

        if mode == "label":
            img = img.astype(np.uint16)

        return img

    @staticmethod
    def min_max_normalization(img: np.ndarray, new_min: float, new_max: float, img_min: float = None, img_max: float = None) -> np.ndarray:
        """
        Min-max-normalizes the given image between new_min and new_max.

        Args:
            img: image to be normalized
            new_min: desired minimum
            new_max: desired maximum
            img_min: min value of the current image. If not given, 
            img_max: max value of the cuttent image

        Returns:
            The normalized image
        """
        if img_min is None: img_min = img.min()
        if img_max is None: img_max = img.max()

        np.add(np.multiply(np.subtract(img, img_min, out=img), (new_max - new_min)/(img_max - img_min), out=img), new_min, out=img)

        return img

    @staticmethod
    def add_poisson_noise(img: np.ndarray, factor: float = 1) -> np.ndarray:
        """
        Adds poisson noise to the given image.

        Args:
            img: image to add poisson noise to
            factor: a factor to change the amount of noise (if its smaller one, there is more noise)

        Returns:
            The given image with added noise
        """
        if factor != 0:
            if factor != 1:
                img *= float(factor)
            rng = np.random.default_rng()
            img = rng.poisson(img).astype(np.float32)
            if factor != 1:
                img /= factor
        return img

    @staticmethod
    def add_gauss_noise(img: np.ndarray, mu: float = 0, sigma: float = 0.01, absolute: bool = False) -> np.ndarray:
        """
        Adds gaussian noise to the given image.

        Args:
            img: image to add the noise to
            mu: mu of the distribution
            sigma: sigma of the distribution
            absolute: if set, the absolute values of the noise are used

        Returns:
            The given image with added noise
        """
        if sigma != 0:
            normal_distribution = np.random.normal(mu, sigma, img.shape)
            if absolute:
                np.abs(normal_distribution, out=normal_distribution)
            img += normal_distribution
        return img

    def run(self, image: np.ndarray, label: np.ndarray, mask_num: int, img_num: int, verbose: bool = False,
            save_interim_images: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates the camera acquisition for the given image.

        Args:
            image: image to process
            label: label data to the given image
            mask_num: number of the current mask used for placement of nuclei (is used for saving the images when save_interim_images is set)
            img_num: number of the current image (is used for saving the images when save_interim_images is set)
            verbose: if set, the command line output is more verbose. If verbose==2, required times for substeps are displayed.
            save_interim_images: if set, all interim results are saved

        Returns:
            The processed image and label
        """
        if verbose:
            s_CAS = time.time()
            print("\n--#--#--#-- Camera Aquisition Simulation --#--#--#--")
        

        ## Downsampling
        if self.px_size_img != self.px_size_desired:
            if verbose:
                print("Image downsampling ...")
                s_CAS_downs = time.time()
            
            image = self.downsample(image, self.px_size_img, self.px_size_desired, mode="image")
            label = self.downsample(label, self.px_size_img, self.px_size_desired, mode="label")
            
            if verbose == 2:
                d_CAS_downs = time.time() - s_CAS_downs; print(f"Image downsampling required {d_CAS_downs:.1f} s -> {d_CAS_downs/60:.1f} min")


        ## Exposure adjustment
        if verbose:
            print("Exposure adjustment ...")
            s_CAS_expo = time.time()

        image = CameraAcquisitionSimulation.min_max_normalization(image, 0, 255 * self.exposure_percentage, img_min=0, img_max=np.quantile(image[label>0], 0.9999))

        if verbose == 2:
                d_CAS_expo = time.time() - s_CAS_expo; print(f"Exposure adjustment required {d_CAS_expo:.1f} s -> {d_CAS_expo/60:.1f} min")


        ## Noise simulation
        if verbose:
            print("Noise simulation ...")
            s_CAS_noise = time.time()
        
        image_iteration = list()
        for n in range(0, self.num_of_acquisition_iterations):
            if self.num_of_acquisition_iterations > 1:
                image_n = np.copy(image)
            else:
                image_n = image

            # adding the dark current signal to the image (simplified as a baseline instead of current * time)
            image_n = self.min_max_normalization(image_n, self.baseline, image.max(), img_min=0)
            if save_interim_images and n == 0:
                tiff.imwrite(os.path.join(self.path_output, "{:03d}_{:03d}_min_max_norm.tif".format(mask_num, img_num)), image_n)
            image_n = self.add_poisson_noise(image_n, factor=self.noise_poisson_factor)
            if save_interim_images and n == 0:
                tiff.imwrite(os.path.join(self.path_output, "{:03d}_{:03d}_poisson.tif".format(mask_num, img_num)), image_n)

            image_n = self.add_gauss_noise(image_n, mu=0, sigma=self.noise_gauss)
            if save_interim_images and n == 0:
                tiff.imwrite(os.path.join(self.path_output, "{:03d}_{:03d}_gauss_1.tif".format(mask_num, img_num)), image_n)

            image_iteration.append(self.add_gauss_noise(image_n, mu=0, sigma=self.noise_gauss_absolute, absolute=True))
            if save_interim_images and n == 0:
                tiff.imwrite(os.path.join(self.path_output, "{:03d}_{:03d}_gauss_2.tif".format(mask_num, img_num)), image_iteration[n])

        if self.num_of_acquisition_iterations > 1:
            image = np.mean(np.array(image_iteration), axis=0)
        else:
            image = image_iteration[0]

        image = np.clip(np.round(image, out=image), 0, 255, out=image).astype(np.uint8)

        if verbose == 2:
            d_CAS_noise = time.time() - s_CAS_noise; print(f"Noise simulation required {d_CAS_noise:.1f} s -> {d_CAS_noise/60:.1f} min")


        if verbose:
            d_CAS = time.time() - s_CAS
            print(f"Camera Aquisition Simulation completed")
            print(f"Time required: {d_CAS:.1f} s -> {d_CAS/60:.1f} min")
        
        return image, label