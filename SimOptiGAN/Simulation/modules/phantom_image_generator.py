import os.path
from random import randint, random
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import skimage.measure
import tifffile as tiff
from scipy.ndimage import rotate
from skimage.transform import rescale
from scipy.ndimage import binary_dilation, generate_binary_structure

from utils import load_data, get_sphere


class PhantomImageGenerator:
    path_phantom_folders: Tuple[str]
    phantom_folders_prop: Tuple[float]
    path_spheroid_mask: str
    shape: Tuple[int, int, int]
    z_append: int
    px_size_phantoms: Tuple[float, float, float]
    px_size_sim_img: Tuple[float, float, float]
    max_overlap: float
    min_dist: int
    break_criterion_positions: int
    break_criterion_objects: int

    path_fragment_folder: str
    fragment_probability: str

    num_of_subcultures: int
    subculture_radius_xy: int
    subculture_radius_z: int
    input_subculture_folder: str
    px_size_subculture_phantoms: Tuple[float, float, float]
    subculture_min_dist: int

    images_folder: str
    labels_folder: str

    phantom_num: int

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
        self.path_phantom_folders = params["path_phantom_folders"]
        self.phantom_folders_prop = params["phantom_folders_prop"]
        self.path_spheroid_mask = params["path_spheroid_mask"]
        self.shape = params["shape"]
        self.z_append = params["z_append"]
        self.px_size_phantoms = params["px_size_phantoms"]
        self.px_size_sim_img = params["px_size_sim_img"]
        self.max_overlap = params["max_overlap"]
        self.min_dist = params["min_dist"]
        self.break_criterion_positions = params["break_criterion_positions"]
        self.break_criterion_objects = params["break_criterion_objects"]

        self.path_fragment_folder = params["path_fragment_folder"]
        self.fragment_probability = params["fragment_probability"]

        self.num_of_subcultures = params["num_of_subcultures"]
        self.subculture_radius_xy = params["subculture_radius_xy"]
        self.subculture_radius_z = params["subculture_radius_z"]
        self.input_subculture_folder = params["input_subculture_folder"]
        self.px_size_subculture_phantoms = params["px_size_subculture_phantoms"]
        self.subculture_min_dist = params["subculture_min_dist"]

        self.images_folder = params["images_folder"]
        self.labels_folder = params["labels_folder"]

    @staticmethod
    def get_phantoms_with_labels(folders: List[str], px_size_phantoms: Tuple[float, float, float],
                                 px_size_sim_img: Tuple[float, float, float]) -> Tuple[list, list]:
        """
        Opens all phantoms in the given folder and resizes them to the given pixel size of the simulated image.

        Args:
            folder: Folder list, that contains all phantoms and their labels. 
                If single folder (str) instead of list is passed, images and labels are returned directly.
            px_size_phantoms: pixel/voxel size of the phantoms
            px_size_sim_img: pixel/voxel size of the simulated image

        Returns:
            Tuple of lists with the first list being the phantoms and the second list being the labels
        """
        
        images_list = list()
        labels_list = list()

        for folder in folders:
            images = list()
            labels = list()

            scale = np.asarray(px_size_phantoms) / np.asarray(px_size_sim_img)

            image_files = [os.path.join(folder, f) for f in os.listdir(folder)
                        if os.path.isfile(os.path.join(folder, f))
                        and f.split(".")[-1] == "tif"
                        and f.count("_label") == 0]

            for image_file in image_files:
                img = load_data(image_file).astype(np.float32)
                label = load_data(image_file[0:-4] + "_label.tif").astype(np.float32)

                img = rescale(img, (scale[0], scale[1], scale[2]), order=1, channel_axis=None, preserve_range=True,
                            anti_aliasing=False)
                label = rescale(label, (scale[0], scale[1], scale[2]), order=1, channel_axis=None, preserve_range=True,
                                anti_aliasing=False)
                label[label > 0] = 1
                # do not labels if they show no foreground after rescaling
                if np.count_nonzero(label) != 0:
                    images.append(img)
                    labels.append(np.clip(label,0,255).astype(np.uint8))
  
            images_list.append(images)
            labels_list.append(labels)

        return images_list, labels_list

    @staticmethod
    def rotate_phantom(img: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotates an image (phantom) and the corresponding label by a random angle between -45 and 45 degree.

        Args:
            img: image of a phantom to be rotated
            label: corresponding label

        Returns:
            The rotated image and label
        """
        angle = randint(-45, 45)

        img = rotate(img, angle=angle, axes=(1, 2), reshape=True)

        label = np.clip(rotate(label.astype(np.float32), angle=angle, order=0, axes=(1, 2), reshape=True, prefilter=False),0,255).astype(np.uint8)

        return img, label

    @staticmethod
    def get_dilated_phantom(phantom_label: np.ndarray, structure: np.ndarray, min_dist: int) -> np.ndarray:
        """
        Dilates a phantom label by the given structures min_dist times

        Args:
            phantom_label: phantom label to be dilated
            structure: structure to be used for the dilation
            min_dist: number of iterations

        Returns:
            The dilated label
        """
        label = np.pad(phantom_label, pad_width=min_dist + 1)
        binary_dilation(label, structure=structure, iterations=min_dist, output=label)
        return label

    @staticmethod
    def get_region_props(label: np.ndarray) -> dict:
        """
        Returns all necessary region properties of a label necessary for placing a phantom in the phantom image

        Args:
            label: label to get the properties for

        Returns:
            Dictionary of the necessary region properties
        """
        props = dict()

        label = skimage.measure.label(label)
        # 0 because there should be just one cell on each image
        sk_props = skimage.measure.regionprops(label)[0]

        props["area"] = sk_props.area
        props["label_coords"] = sk_props.coords
        label_centroid = np.round(np.asarray(sk_props.centroid)).astype(np.uint16)
        props["coords_centered"] = props["label_coords"] - label_centroid

        label_dim_max_z = max(np.max(props["coords_centered"][:, 0]), -np.min(props["coords_centered"][:, 0]))
        label_dim_max_y = max(np.max(props["coords_centered"][:, 1]), -np.min(props["coords_centered"][:, 1]))
        label_dim_max_x = max(np.max(props["coords_centered"][:, 2]), -np.min(props["coords_centered"][:, 2]))

        props["label_dim_max"] = np.asarray([label_dim_max_z, label_dim_max_y, label_dim_max_x])

        return props

    @staticmethod
    def gen_fragmented_phantom(phantom_label: np.ndarray, phantom_volume: int, fragment_images: List[np.ndarray], fragment_labels: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Generates a fragmented phantom.

        Args:
            phantom_label: mask which defines the shape of the fragmented phantom
            phantom_volume: volume of the phantom label
            fragment_images: list of images to use as fragments
            fragment_labels: list of corresponding labels

        Returns:
            phantom image, it's label and the centered coords
        """
        phantom_label = np.copy(phantom_label)
        # phantom_label = rescale(phantom_label, 2, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        phantom_image = np.zeros_like(phantom_label, dtype=np.float32)
        props = PhantomImageGenerator.get_region_props(phantom_label)
        frag_vol, frag_vol_prev = 0, 0
        frag_abort_counter = 0
        while frag_vol < phantom_volume*0.05 and frag_abort_counter <= 100:
            nuclei_nr = randint(0, len(fragment_images) - 1)
            fragment_image = fragment_images[nuclei_nr]
            fragment_label = fragment_labels[nuclei_nr]
            # Slice to avoid the case, that the coordinates of the fragments are larger than the shape of the phantom label.
            valid_sl = tuple([slice(0, end_p - end_f + 1) for end_p, end_f in zip(phantom_label.shape, fragment_label.shape)])
            # Random position in the valid region of the label mask
            pos = np.unravel_index(np.random.choice(np.flatnonzero(phantom_label[valid_sl])), phantom_label[valid_sl].shape)
            # Convert local fragment coords to the global phantom coords
            coords_frag_f = np.nonzero(fragment_label)
            coords_frag_p = tuple((np.asarray(coords_frag_f).T + pos).T)
            if np.all(phantom_label[coords_frag_p]>=1):
                phantom_image[coords_frag_p] = fragment_image[coords_frag_f]
                # Count only areas not covered by previous fragment placements
                frag_vol = np.count_nonzero(phantom_label==2)
                phantom_label[coords_frag_p] = 2
            if (frag_vol-frag_vol_prev)/(frag_vol+1e-5) <= 0.01:
                frag_abort_counter += 1
            else:
                frag_abort_counter = 0
            frag_vol_prev = frag_vol
        phantom_label = phantom_label > 1 # Only use the placed fragments as label

        return phantom_image, phantom_label, props

    def place_subculture_phantoms_in_image(self, image_sim: np.ndarray, label_sim: np.ndarray) -> Tuple[np.ndarray,
                                                                                                        np.ndarray]:
        """
        Places a subculture in the given image.

        Args:
            image_sim: phantom image to place the subculture in
            label_sim: label to the phantom image

        Returns:
            phantom image and it's label with the added subculture
        """
        phantom_images, phantom_labels = self.get_phantoms_with_labels([self.input_subculture_folder],
                                                                       self.px_size_subculture_phantoms,
                                                                       self.px_size_sim_img)
        z, y, x = self.shape
        mask = np.zeros_like(label_sim)

        spheroid_mask = load_data(self.path_spheroid_mask)

        r_xy = self.subculture_radius_xy
        r_z = self.subculture_radius_z
        subculture_mask = get_sphere(r_z, r_xy, r_xy, z_step=int(r_xy / r_z))
        z_mask, y_mask, x_mask = subculture_mask.shape

        count_fail = 0

        while True:
            z_corner = randint(0, z - z_mask - 10)  # Don't cut the subculture
            y_corner = randint(0, y - y_mask)
            x_corner = randint(0, x - x_mask)

            overlap_mask = np.count_nonzero(np.bitwise_and(spheroid_mask[z_corner:z_corner + z_mask,
                                                           y_corner:y_corner + y_mask,
                                                           x_corner:x_corner + x_mask],
                                                           subculture_mask)) / np.count_nonzero(subculture_mask)
            overlap_image = np.count_nonzero(np.bitwise_and(label_sim[z_corner:z_corner + z_mask,
                                                            y_corner:y_corner + y_mask,
                                                            x_corner:x_corner + x_mask],
                                                            subculture_mask)) / np.count_nonzero(subculture_mask)

            if overlap_mask > 0.9 and overlap_image <= self.max_overlap:
                mask[z_corner:z_corner + z_mask,
                     y_corner:y_corner + y_mask,
                     x_corner:x_corner + x_mask] = subculture_mask
                return self.place_objects_in_image(image_sim, label_sim, phantom_images, phantom_labels, mask,
                                                   self.max_overlap, self.subculture_min_dist)

            count_fail += 1
            if count_fail > 100:
                print("Warning: could not find a place for the subculture")
                break

        return image_sim, label_sim

    def place_phantoms_in_image(self, image_sim: np.ndarray, label_sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Places phantoms in the given phantom image

        Args:
            image_sim: phantom image to place the phantoms in
            label_sim: label to the phantom image

        Returns:
            phantom image and it's label with the added phantoms
        """
        phantom_images, phantom_labels = self.get_phantoms_with_labels(self.path_phantom_folders, self.px_size_phantoms,
                                                                       self.px_size_sim_img)
        
        if self.path_fragment_folder is not None and self.path_fragment_folder != '':
            fragment_images, fragment_labels = self.get_phantoms_with_labels([self.path_fragment_folder], self.px_size_phantoms,
                                                                        self.px_size_sim_img)
            fragment_images, fragment_labels = fragment_images[0], fragment_labels[0]
            fragment_images = [f_img * 10 for f_img in fragment_images] # make the fragments 10-times brighter
        else:
            fragment_images, fragment_labels = None, None

        spheroid_mask = load_data(self.path_spheroid_mask)

        return self.place_objects_in_image(image_sim, label_sim, phantom_images, phantom_labels, spheroid_mask,
                                           self.max_overlap, self.min_dist, self.phantom_folders_prop, fragment_images, fragment_labels)

    def place_objects_in_image(self, image_sim: np.ndarray, label_sim: np.ndarray, phantom_images: List[List[np.ndarray]],
                               phantom_labels: List[List[np.ndarray]], mask: np.ndarray, max_overlap: float, min_dist: int,
                               phantom_type_props: List[float]=None,
                               fragment_images: np.ndarray=None, fragment_labels: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Places objects/phantoms in an image inside a given mask.
        The mask can be a subculture mask or the mask of the whole cell culture.

        Args:
            image_sim: phantom image to place the phantoms in
            label_sim: label to the phantom image
            phantom_images: list of phantoms
            phantom_labels: list of the phantom labels
            mask: mask to place the phantoms in
            max_overlap: maximum overlap between phantoms
            min_dist: minimum overlap between phantoms

        Returns:
            phantom image and it's label with the added phantoms
        """

        dilation_structure = generate_binary_structure(3, 1)

        args = np.argwhere(mask.max(axis=(1,2)))
        min_z = args.min()
        max_z = args.max()
        args = np.argwhere(mask.max(axis=(0,2)))
        min_y = args.min()
        max_y = args.max()
        args = np.argwhere(mask.max(axis=(0,1)))
        min_x = args.min()
        max_x = args.max()
        del args

        phantom_types = []
        n_folder = len(phantom_images)
        if phantom_type_props is not None:
            phantom_type_props = np.asarray(phantom_type_props) / np.sum(phantom_type_props)
            if len(phantom_type_props) != n_folder:
                raise ValueError('Number of folders and number of propabilities do not match!')


        objects_tried = 0
        fragment_ind_list = []
        while objects_tried < self.break_criterion_objects:
            objects_tried += 1
            
            # random selection of the phantom to be placed in the image
            if n_folder == 1:
                folder_ind = 0
            elif phantom_type_props is None:
                folder_ind = randint(0, n_folder-1)
            else:
                folder_ind = np.random.choice(n_folder, p=phantom_type_props)
            
            nuclei_nr = randint(0, len(phantom_images[folder_ind]) - 1)
            phantom_image = phantom_images[folder_ind][nuclei_nr]
            phantom_label = phantom_labels[folder_ind][nuclei_nr]

            # augmentation (rotation)
            if random() > 0.5:
                phantom_image, phantom_label = PhantomImageGenerator.rotate_phantom(phantom_image, phantom_label)

            props = PhantomImageGenerator.get_region_props(phantom_label)
            
            # Fragmented phantom
            fragment = False
            if random() < self.fragment_probability and fragment_images is not None and fragment_labels is not None:
                fragment = True # Needed for later label removal
                phantom_image, phantom_label, props = PhantomImageGenerator.gen_fragmented_phantom(phantom_label, props["area"], 
                                                                                                   fragment_images, fragment_labels)

            coords_centered = props["coords_centered"]
            if min_dist > 0:
                # if a minimum distance is given a dilated label is used, to check overlap with other phantoms
                label_dilated = PhantomImageGenerator.get_dilated_phantom(phantom_label, dilation_structure, min_dist)
                props_dilated = PhantomImageGenerator.get_region_props(label_dilated)
                coords_centered_dilated = props_dilated["coords_centered"]
                label_dim_max_dilated = props_dilated["label_dim_max"]
            else:
                coords_centered_dilated = coords_centered
                label_dim_max_dilated = props["label_dim_max"]

            # augmentation (mirroring)
            if random() > 0.5: # rot 90° z-axis (switch x and y axis)
                coords_centered = coords_centered[:, [0, 2, 1]]
                coords_centered_dilated = coords_centered_dilated[:, [0, 2, 1]]
                label_dim_max_dilated = label_dim_max_dilated[[0, 2, 1]]
            if random() > 0.5: # flip z-x-plane (up/down)
                coords_centered[:, 1] = -coords_centered[:, 1]
                coords_centered_dilated[:, 1] = -coords_centered_dilated[:, 1]
            if random() > 0.5: # flip z-y-plane (left/right)
                coords_centered[:, 2] = -coords_centered[:, 2]
                coords_centered_dilated[:, 2] = -coords_centered_dilated[:, 2]
            if random() > 0.5: # flip y-x-plane
                coords_centered[:, 0] = -coords_centered[:, 0]
                coords_centered_dilated[:, 0] = -coords_centered_dilated[:, 0]

            pos_tried = 0
            while pos_tried < self.break_criterion_positions:
                pos_tried += 1
                # random position to place the object
                while True: # Sample until position is valid
                    value = [randint(max(label_dim_max_dilated[0], min_z),
                                    min(image_sim.shape[0] - label_dim_max_dilated[0] - 1, max_z)),
                            randint(max(label_dim_max_dilated[1], min_y),
                                    min(image_sim.shape[1] - label_dim_max_dilated[1] - 1, max_y)),
                            randint(max(label_dim_max_dilated[2], min_x),
                                    min(image_sim.shape[2] - label_dim_max_dilated[2] - 1, max_x))]
                    if mask[value[0], value[1], value[2]] > 0 and label_sim[value[0], value[1], value[2]] == 0:
                        break

                coords = np.add(coords_centered, value)
                coords_dilated = np.add(coords_centered_dilated, value)

                # calculate overlap to existing objects
                overlap = np.count_nonzero(label_sim[coords_dilated[:, 0], coords_dilated[:, 1], coords_dilated[:, 2]])

                # calculate overlap with the given mask (100% of phantom should be inside for later brightness reduction)
                overlap_with_mask = np.count_nonzero(mask[coords[:, 0], coords[:, 1], coords[:, 2]])

                if overlap / props["area"] <= max_overlap and overlap_with_mask == props["area"]:
                    # Place phantom
                    label_sim[coords[:, 0], coords[:, 1], coords[:, 2]] = self.phantom_num
                    image_sim[coords[:, 0], coords[:, 1], coords[:, 2]] = phantom_image[
                                                                            props["label_coords"][:, 0],
                                                                            props["label_coords"][:, 1],
                                                                            props["label_coords"][:, 2]
                                                                            ]
                    if fragment:
                        fragment_ind_list.append(self.phantom_num)
                    phantom_types.append(folder_ind if not fragment else -1)
                    self.phantom_num += 1
                    objects_tried = 0
                    break # -> next object

        # Remove fragment labels
        for fragment_ind in fragment_ind_list:
            label_sim[label_sim==fragment_ind] = 0
        
        return image_sim, label_sim, phantom_types

    def run(self, mask_num: int, img_num: int, verbose: bool = False, save_interim_images: bool = False) -> Tuple[np.ndarray,
                                                                                                   np.ndarray]:
        """
        Generates a phantom image using the given parameters

        Args:
            mask_num: number of the current mask used for placement of nuclei (is used for saving the images when save_interim_images is set)
            img_num: number of the current image (is used for saving the images when save_interim_images is set)
            verbose: if set, there is command line output about the process. If verbose==2, required times for substeps are displayed.
            save_interim_images: if set, all interim results are saved

        Returns:
            The generated image and it's label
        """
        if verbose:
            s_PIG = time.time()
            print("\n--#--#--#-- Phantom Image Generator --#--#--#--")

        self.phantom_num = 1
        # create the new empty images
        image_sim = np.zeros(self.shape, dtype=np.float32)
        label_sim = np.zeros(self.shape, dtype=np.uint16)
        

        ### Subculture generation
        if self.num_of_subcultures > 0:
            if verbose:
                print("Placing subcultures ...")
            s_PIG_subc = time.time()

            for n in range(0, self.num_of_subcultures):
                image_sim, label_sim = self.place_subculture_phantoms_in_image(image_sim, label_sim)

            if verbose == 2:
                d_PIG_subc = time.time() - s_PIG_subc; print(f"Placing subcultures required {d_PIG_subc:.1f} s -> {d_PIG_subc/60:.1f} min")
            if save_interim_images:
                tiff.imwrite(os.path.join(self.images_folder, "{:03d}_{:03d}_subcultures.tif".format(mask_num, img_num)), image_sim)
                tiff.imwrite(os.path.join(self.labels_folder, "{:03d}_{:03d}_subcultures_label.tif".format(mask_num, img_num)), label_sim)


        ### Phantom placement
        if verbose:
            print("Placing phantoms ...")
            s_PIG_phant = time.time()

        image_sim, label_sim, phantom_types = self.place_phantoms_in_image(image_sim, label_sim)
        
        if verbose == 2:
            d_PIG_phant = time.time() - s_PIG_phant; print(f"Placing phantoms required {d_PIG_phant:.1f} s -> {d_PIG_phant/60:.1f} min")
        if save_interim_images:
            tiff.imwrite(os.path.join(self.images_folder, "{:03d}_{:03d}_prototype.tif".format(mask_num, img_num)), image_sim)


        if verbose:
            d_PIG = time.time() - s_PIG
            print(f"Phantom Image Generator completed")
            print(f"Time required: {d_PIG:.1f} s -> {d_PIG/60:.1f} min")

        return image_sim, label_sim, phantom_types