import numpy as np
import tifffile as tiff
from skimage import measure
import scipy.ndimage as ndimage


def get_biggest_object(objects: np.ndarray) -> np.ndarray:
    """
    Removes all objects other than the biggest one.

    Args:
        objects: binary image

    Returns:
        Image, with just the biggest object left
    """
    out = np.copy(objects.astype(np.uint8))

    out_mask = measure.label(out)
    props = measure.regionprops(out_mask)
    biggest = 0
    biggest_area = 0

    for n, cur_props in enumerate(props):
        elements = cur_props.area

        if elements > biggest_area:
            biggest_area = elements
            biggest = n

    for n, cur_props in enumerate(props):
        if n != biggest:
            label_coords = cur_props.coords
            out[label_coords[:, 0], label_coords[:, 1], label_coords[:, 2]] = 0

    return out


def fill_holes(image: np.ndarray) -> np.ndarray:
    """
    Fills holes in the given image.

    Args:
        image: binary image

    Returns:
        The image with filled holes
    """
    out = np.copy(image)

    img_inv = np.copy(image)
    img_inv[image == 0] = 1
    img_inv[image != 0] = 0

    img_inv_mask = measure.label(img_inv)
    props = measure.regionprops(img_inv_mask)
    biggest = 0
    biggest_area = 0

    for n, cur_props in enumerate(props):
        elements = cur_props.area

        if elements > biggest_area:
            biggest_area = elements
            biggest = n

    for n, cur_props in enumerate(props):
        if n != biggest:
            label_coords = cur_props.coords
            out[label_coords[:, 0], label_coords[:, 1], label_coords[:, 2]] = 1

    return out


def extract_spheroid_mask(image: np.ndarray) -> np.ndarray:
    """
    Tries to extract a mask for the cell culture by over segmentation

    Args:
        image: microscopy image of a cell culture

    Returns:
        binary image containing the cell culture mask
    """
    image = ndimage.gaussian_filter(image, sigma=5)
    image[image < np.mean(image)] = 0

    structure = np.array([[[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]],
                          [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]],
                          [[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]]])

    structure2 = np.array([[[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0]],
                           [[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]],
                           [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
                           [[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]],
                           [[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0]]])

    print("binary opening")
    mask = ndimage.binary_opening(image, structure=structure, iterations=1).astype(np.uint8)
    image[mask == 0] = 0

    print("biggest object")
    image[image > 0] = 1
    image = get_biggest_object(image)

    print("fill holes")
    image[image > 0] = 1
    image = fill_holes(image)

    print("binary closing")
    for n in range(0, 10):
        mask = ndimage.binary_dilation(image, structure=structure2, iterations=1).astype(np.uint8)
        mask = ndimage.binary_erosion(mask, structure=structure2, iterations=1).astype(np.uint8)
    image[mask > 0] = 1

    return image


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser("Tries to extract spheroid masks out of microscopy images in the given folder")
    parser.add_argument("image_folder", type=str,
                        help="The folder the microscopy images are located in")
    parser.add_argument("out_folder", type=str,
                        help="The folder to save the extracted masks to")
    args = parser.parse_args()

    image_folder = vars(args)["image_folder"]
    out_folder = vars(args)["out_folder"]

    if not os.path.isdir(image_folder):
        raise AttributeError("The given image folder is no valid directory")
    if not os.path.isdir(out_folder):
        raise AttributeError("The given out folder is no valid directory")

    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if os.path.isfile(os.path.join(image_folder, f))
                   and f.endswith(".tif")]

    n = -1
    for n, image_file in enumerate(image_files):
        print("begin with {}. image".format(n+1))
        img = tiff.imread(image_file)
        spheroid_mask = extract_spheroid_mask(img)
        tiff.imwrite(os.path.join(out_folder, "spheroid_mask_{}.tif".format(n)), spheroid_mask * 255)
        print("{}. mask extracted ({} to go)".format(n+1, len(image_files)-n-1))
        if len(image_files)-n-1 > 0:
            print("-----------------------------------------------------------")

    if n == -1:
        print("Warning: There were no image files to extract a mask from (just .tif files are supported)")
