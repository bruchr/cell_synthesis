from pathlib import Path

import numpy as np
from scipy import ndimage
import tifffile as tiff


def min_max_normalization(img, min_value=None, max_value=None, noise_level=None):
    """ Min-max-normalization for (volumetric) uint images.

    :param img: Image with shape [depth, height, width], [height, width] or [height, width, color channel].
        :type img:
    :param min_value: Minimum value for the normalization. All values below this value are clipped
        :type min_value: int
    :param max_value: Maximum value for the normalization. All values above this value are clipped.
        :type max_value: int
    :param noise_level: If (max_value - min_value) < noise_level, the image, which probably only contains noise, is
        normalized to the range of the used data type (appears dark instead of bright).
    :return: Normalized (volumetric) image (float32)
    """

    if max_value is None:  # Get new maximum value for the normalization. Filter the image to avoid hot/cold pixels

        if len(img.shape) == 3:  # Check for volumetric or color images

            if img.shape[-1] == 3:  # rgb image

                raise Exception('Not implemented yet')

            else:  # Volumetric image

                img_filtered = ndimage.median_filter(input=img, size=(1, 3, 3))

        else:  # 2d grayscale image

            img_filtered = ndimage.median_filter(input=img, size=(3, 3))

        max_value = img_filtered.max()

        if min_value is None:  # Get new minimum value

            min_value = img_filtered.min()

    if noise_level is not None:  # Avoid the normalization of images, that only contain noise

        if (max_value - min_value) < noise_level:
            max_value = np.iinfo(img.dtype).max
            min_value = 0

    # Clip image to filter hot and cold pixels
    img = np.clip(img, min_value, max_value)

    # Apply Min-max-normalization
    img = 2 * (img.astype(np.float32) - min_value) / (max_value - min_value) - 1

    return img.astype(np.float32)


if __name__ == '__main__':
    path_dataset = Path('C:/Users/xx3662/Desktop/Projekte_Daten/Marker_Synthesis/Data/Membrane-Nuclei/cGAN/td_v3')
    path_new_dataset = path_dataset.parent / 'td_v3.1'

    folders_2_process = ['trainA', 'trainB', 'inferenceA', 'inferenceB']

    path_new_dataset.mkdir(parents=True, exist_ok=False)
    with open(path_new_dataset/'info.txt', 'w') as info_file:
        info_file.write(f'Automatically normalized data (-1 to 1) of path:\n{path_dataset}')

    for folder in folders_2_process:
        path_folder = path_dataset / folder
        if not path_folder.exists():
            print(f'No folder with name {folder} found. Skipping...')
            continue
        path_folder_new = path_new_dataset / folder
        path_folder_new.mkdir(parents=False, exist_ok=False)
        for file_path in path_folder.glob('*.tif'):
            file_path_new = path_folder_new / file_path.name
            img = tiff.imread(file_path)
            print(f'Processing: {file_path.relative_to(path_dataset.parent)}')
            img = min_max_normalization(img)
            tiff.imwrite(file_path_new, img)