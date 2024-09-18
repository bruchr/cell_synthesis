from os.path import isfile, isdir, join, exists
from typing import List, Optional

import nrrd
import numpy as np
import skimage.measure
import tifffile as tiff


class PhantomExtractor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def is_valid_image_file(path: str) -> bool:
        """
        Checks if a file exists and if it's a tiff or nrrd file

        Args:
            path: path to the image file

        Returns:
            True if the file exists and is in a supported format, otherwise False
        """
        if not isfile(path):
            return False
        elif path.endswith('.tiff') or path.endswith('.tif') or path.endswith('.nrrd'):
            return True
        else:
            return False

    @staticmethod
    def load_data(path: str) -> np.ndarray:
        """
        Loads an image file either in tiff or nrrd format

        Args:
            path: path to the image file

        Returns:
            A numpy array representing the image data
        """
        if path.endswith('.tiff') or path.endswith('.tif'):
            return tiff.imread(path)
        elif path.endswith('.nrrd'):
            return nrrd.read(path, index_order='C')[0]
        else:
            raise Exception('Only .tiff or .nrrd images can be read.')

    @staticmethod
    def min_max_normalization(img: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
        """
        Min-max-normalizes the given image between new_min and new_max.

        Args:
            img: image to be normalized
            new_min: desired minimum
            new_max: desired maximum

        Returns:
            The normalized image
        """
        img_min = img.min()
        img_max = img.max()

        img_norm = ((img - img_min) / (img_max - img_min)) * (new_max - new_min) + new_min

        return img_norm.astype(np.uint8)

    @staticmethod
    def on_border(im_shape: tuple, bbox: list) -> bool:
        """
        Checks if bbox touches the border of an image and returns a bool value.

        Args:
            im_shape: shape of the image
            bbox: bounding box coordinates as a list

        Returns:
            True if the bounding box touches the image border, otherwise False
        """

        result = False
        if len(bbox) == 6:  # 3D images
            if bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] <= 0:
                result = True
            if bbox[3] >= im_shape[0] or bbox[4] >= im_shape[1] or bbox[5] >= im_shape[2]:
                result = True
        else:  # 2D images
            if bbox[0] <= 0 or bbox[1] <= 0:
                result = True
            if bbox[2] >= im_shape[0] or bbox[3] >= im_shape[1]:
                result = True

        return result

    @staticmethod
    def extract_valid_phantoms_from_files(image: str, label: str, out: Optional[str] = None, prefix: Optional[str] = '') \
            -> List[skimage.measure._regionprops.RegionProperties]:
        """
        Returns a list of region properties for all valid phantoms in the given image

        Args:
            image: path to the image file
            label: path to the label file
            out: optional path to save the valid phantoms and it's labels to

        Returns:
            list of skimage region properties
        """
        return PhantomExtractor.extract_valid_phantoms(PhantomExtractor.load_data(image),
                                                       PhantomExtractor.load_data(label),
                                                       out, prefix)

    @staticmethod
    def extract_valid_phantoms(image: np.ndarray, label: np.ndarray, out: Optional[str] = None, prefix: Optional[str] = '') \
            -> List[skimage.measure._regionprops.RegionProperties]:
        """
        Returns a list of region properties for all valid phantoms in the given image

        Args:
            image: image to find the phantoms in
            label: label to the given image
            out: optional path to save the valid phantoms and it's labels to

        Returns:
            list of skimage region properties
        """
        if out is not None and not isdir(out):
            print("Warning: The given out folder is invalid. No phantoms are saved!")
            out = None

        # Get the properties of the segmented nuclei
        label_orig = label
        # label = skimage.measure.label(label) # uncomment if not every object has its own id
        props = skimage.measure.regionprops(label)

        valid_props = list()  # List for segmented objects that do not lie on the image border
        for nuclei_nr in range(len(props)):
            if not PhantomExtractor.on_border(image.shape, props[nuclei_nr].bbox):
                valid_props.append(props[nuclei_nr])

                if out is not None:
                    naming_offset = 0

                    if prefix != '' and prefix[-1] != '_': prefix = prefix + '_'
                    
                    while True:
                        file_name = join(out, prefix + "phantom_{}.tif".format(str(nuclei_nr + naming_offset)))
                        if exists(file_name):
                            naming_offset += 1
                            continue

                        coords_g = props[nuclei_nr].coords
                        coords_l = coords_g - coords_g.min(axis=0)

                        img_out = np.zeros(coords_l.max(axis=0)+1, dtype=image.dtype)
                        label_out = np.zeros(coords_l.max(axis=0)+1, dtype=np.uint16)

                        img_out[tuple(coords_l.T)] = image[tuple(coords_g.T)]
                        img_out = PhantomExtractor.min_max_normalization(img_out, 0, 255)
                        label_out[tuple(coords_l.T)] = label_orig[tuple(coords_g.T)]


                        tiff.imwrite(file_name, img_out)
                        tiff.imwrite(file_name[0:-4] + "_label.tif", label_out)
                        break

        return valid_props


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Extract Phantoms out of a labeled microscopy image")
    parser.add_argument("image", type=str,
                        help="The microscopy image to use")
    parser.add_argument("label", type=str,
                        help="The label file to use")
    parser.add_argument("out_folder", type=str,
                        help="The folder to save the extracted phantoms to")
    parser.add_argument("prefix", type=str, default='',
                        help="String added to filename")
    args = parser.parse_args()

    image_file = vars(args)["image"]
    label_file = vars(args)["label"]
    out_folder = vars(args)["out_folder"]
    prefix = vars(args)["prefix"]

    if not PhantomExtractor.is_valid_image_file(image_file):
        raise AttributeError("The given image is no valid image file (.tif, .tiff or .nrrd are supported)")
    if not PhantomExtractor.is_valid_image_file(label_file):
        raise AttributeError("The given label is no valid image file (.tif, .tiff or .nrrd are supported)")
    if not isdir(out_folder):
        raise AttributeError("The given out folder is no valid directory")

    PhantomExtractor.extract_valid_phantoms_from_files(image_file, label_file, out_folder, prefix)
