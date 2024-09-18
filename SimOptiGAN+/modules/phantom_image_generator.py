import os.path
from random import randint, random
import time
from typing import List, Tuple, Dict, Any

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, generate_binary_structure, rotate
import skimage.measure
from skimage.transform import rescale
import tifffile as tiff
from tqdm import tqdm

from utils import load_data, get_sphere


class PhantomImageGenerator:
    path_phantom_folders: Tuple[str]
    phantom_folders_prop: Tuple[float]
    path_spheroid_mask: str
    shape: Tuple[int, int, int]
    z_append: int
    px_size_phantoms: Tuple[float, float, float]
    px_size_mask: Tuple[float, float, float]
    px_size_sim_img: Tuple[float, float, float]
    max_overlap: float
    break_criterion_positions: int
    break_criterion_objects: int

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
        self.px_size_mask = params["px_size_mask"]
        self.px_size_sim_img = params["px_size_sim_img"]
        self.vol_ratio = params["volume_ratio_phantom"]
        self.max_overlap = params["max_overlap"]
        self.break_criterion_positions = params["break_criterion_positions"]
        self.break_criterion_objects = params["break_criterion_objects"]

        self.images_folder = params["images_folder"]
        self.labels_folder = params["labels_folder"]

    @staticmethod
    def get_phantoms_with_labels(folders: List[str], px_size_phantoms: Tuple[float, float, float],
                                 px_size_sim_img: Tuple[float, float, float]) -> Tuple[list, list, list]:
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
        folder_ind_list = list()

        for ind_folder, folder in enumerate(folders):

            # scale = np.asarray(px_size_phantoms) / np.asarray(px_size_sim_img)

            image_files = [os.path.join(folder, f) for f in os.listdir(folder)
                        if os.path.isfile(os.path.join(folder, f))
                        and f.split(".")[-1] == "tif"
                        and f.count("_label") == 0]

            for image_file in image_files:
                img = load_data(image_file).astype(np.float32)
                label = load_data(image_file[0:-4] + "_label.tif").astype(np.float32)

                label[label > 0] = 1
                # do not labels if they show no foreground after rescaling
                if np.count_nonzero(label) != 0:
                    images_list.append(img)
                    labels_list.append(np.clip(label,0,255).astype(np.uint8))
                    folder_ind_list.append(ind_folder)

        return images_list, labels_list, folder_ind_list

    @staticmethod
    def scale_phantom(img: np.ndarray, label: np.ndarray, scale=Tuple[float]) -> Tuple[np.ndarray, np.ndarray]:
        img = rescale(img, (scale[0], scale[1], scale[2]), order=1, channel_axis=None, preserve_range=True,
                      anti_aliasing=False)
        # label = rescale(label, (scale[0], scale[1], scale[2]), order=1, channel_axis=None, preserve_range=True,
        #                 anti_aliasing=False)
        label = rescale(label, (scale[0], scale[1], scale[2]), order=0, channel_axis=None, preserve_range=True,
                        anti_aliasing=False)
        label[label > 0] = 1
        label = np.clip(label,0,255).astype(np.uint8)

        return img, label

    @staticmethod
    def rotate_phantom(img: np.ndarray, label: np.ndarray, angle:float=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotates an image (phantom) and the corresponding label by a random angle between -45 and 45 degree.

        Args:
            img: image of a phantom to be rotated
            label: corresponding label

        Returns:
            The rotated image and label
        """
        if angle is None:
            angle = randint(-45, 45)

        img = rotate(img, angle=angle, axes=(1, 2), reshape=True)

        label = np.clip(rotate(label.astype(np.float32), angle=angle, order=0, axes=(1, 2), reshape=True, prefilter=False),0,255).astype(np.uint8)

        return img, label

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
        props["label_centroid"] = np.round(np.asarray(sk_props.centroid)).astype(np.uint16)
        props["coords_centered"] = props["label_coords"] - props["label_centroid"]

        label_dim_max_z = max(np.max(props["coords_centered"][:, 0]), -np.min(props["coords_centered"][:, 0]))
        label_dim_max_y = max(np.max(props["coords_centered"][:, 1]), -np.min(props["coords_centered"][:, 1]))
        label_dim_max_x = max(np.max(props["coords_centered"][:, 2]), -np.min(props["coords_centered"][:, 2]))

        props["label_dim_max"] = np.asarray([label_dim_max_z, label_dim_max_y, label_dim_max_x])

        return props
    

    def place_phantoms_in_image(self, image_sim: np.ndarray, label_sim: np.ndarray, mask_img_num:Tuple[int]=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Places phantoms in the given phantom image

        Args:
            image_sim: phantom image to place the phantoms in
            label_sim: label to the phantom image

        Returns:
            phantom image and it's label with the added phantoms
        """
        phantom_images, phantom_labels, folder_ind_list = self.get_phantoms_with_labels(self.path_phantom_folders, self.px_size_phantoms,
                                                                       self.px_size_sim_img)

        spheroid_mask = load_data(self.path_spheroid_mask)

        return self.place_objects_in_image(image_sim, label_sim, phantom_images, phantom_labels, folder_ind_list, spheroid_mask,
                                           self.max_overlap, self.phantom_folders_prop)

    def place_objects_in_image(self, image_sim: np.ndarray, label_sim: np.ndarray, phantom_images: List[np.ndarray],
                               phantom_labels: List[np.ndarray], folder_ind_list: List[int], mask: np.ndarray,
                               max_overlap: float, phantom_type_props: List[float]=None) -> Tuple[np.ndarray, np.ndarray]:
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

        Returns:
            phantom image and it's label with the added phantoms
        """

        phantom_types = []

        cell_props = skimage.measure.regionprops(mask)
        mean_vol = np.mean([cell_prop.area for cell_prop in cell_props])
        cell_props = [cell_prop for cell_prop in cell_props if cell_prop.area > 5 and cell_prop.area < mean_vol*20] # Cells with only one voxel will lead to errors in morph calculation!
        cell_morph = self.get_morph_props(cell_props, self.px_size_sim_img)
        nuc_morph = self.get_morph_props([skimage.measure.regionprops(phantom_label)[0] for phantom_label in phantom_labels], self.px_size_phantoms)

        counter_pos_tries = 0
        counter_object_tries = 0
        counter_cell_ids = 0
        counter_aborted = 0
        dc_prop = 0
        dc_rot = 0
        dc_scale = 0
        dc_pos = 0
        nuclei_nr_amount = np.zeros(len(phantom_labels), dtype=np.uint16)

        for ind_cell, cell_prop in enumerate(tqdm(cell_props)):
            counter_cell_ids += 1
            
            object_placed = 0
            s_prop = time.time()
            

            bb = np.asarray(cell_prop.bbox)
            ndim = cell_prop.coords.shape[1]
            c = cell_prop.coords-bb[0:ndim]
            bb_size = (bb[2]-bb[0], bb[3]-bb[1]) if ndim==2 else (bb[3]-bb[0], bb[4]-bb[1], bb[5]-bb[2])
            img_crop = np.zeros(bb_size)
            img_crop[tuple(c.T)] = 1
            propability = distance_transform_edt(img_crop)
            # Normalization based on area of hollow sphere with r1-r2=1
            # The idea is to compensate that more pixel are located at the outer regions.
            prop_rad = propability.max()-propability
            prop_norm = 4/3 * np.pi * (np.power(prop_rad+1, 3) - np.power(prop_rad, 3))
            propability = propability/prop_norm
            propability = propability/propability.sum()

            # match_prop = np.abs(nuc_morph['eccentricity'] - cell_morph['eccentricity'][ind_cell])
            
            # without matching:
            match_prop = np.random.rand(len(nuc_morph['volume_um']))

            d_prop = time.time() - s_prop; dc_prop += d_prop; #print(f'{d_prop:.3f} s needed for matching.')

            objects_tried = 0
            while (object_placed == 0) and (objects_tried < self.break_criterion_objects):
                counter_object_tries += 1
                objects_tried += 1

                # random selection of the phantom to be placed in the image
                nuclei_nr = np.argmin(match_prop)
                match_prop[nuclei_nr] = np.inf

                phantom_image = phantom_images[nuclei_nr]
                phantom_label = phantom_labels[nuclei_nr]

                # augmentation (rotation)
                s_rot = time.time()
                rot_angle = np.rad2deg(nuc_morph['orientation_xy'][nuclei_nr] - cell_morph['orientation_xy'][ind_cell])
                phantom_image, phantom_label = PhantomImageGenerator.rotate_phantom(phantom_image, phantom_label, angle=rot_angle)

                d_rot = time.time() - s_rot; dc_rot += d_rot

                s_scale = time.time()
                # Scale factor for nuclei to match desired cell-nuclei volume ratio
                scale_nuc2cell = ((cell_morph['volume_um'][ind_cell] * self.vol_ratio) / nuc_morph['volume_um'][nuclei_nr])**(1/3)
                # Scale factor to convert from px_phantom to px_sim_image
                scale_phan2sim = np.divide(self.px_size_phantoms, self.px_size_sim_img)
                # XY/Z proportion should be similar
                scale_proportion_cell = (np.mean(bb_size[1:]) * np.mean(self.px_size_sim_img[1:])) / (bb_size[0] * self.px_size_sim_img[0])
                scale_proportion_nuc = (np.mean(phantom_label.shape[1:]) * np.mean(self.px_size_phantoms[1:])) / (phantom_label.shape[0] * self.px_size_phantoms[0])
                scale_proportion_f = scale_proportion_cell / scale_proportion_nuc
                scale_proportion_f_x = (1/scale_proportion_f)**(1/3) # Product of all scales should be one
                scale_proportion = np.divide(1, np.asarray((scale_proportion_f * scale_proportion_f_x, scale_proportion_f_x, scale_proportion_f_x)))

                scale = scale_nuc2cell * scale_phan2sim * scale_proportion
                phantom_image, phantom_label = PhantomImageGenerator.scale_phantom(phantom_image, phantom_label, scale)
                d_scale = time.time() - s_scale; dc_scale += d_scale

                props = PhantomImageGenerator.get_region_props(phantom_label)
                coords_centered = props["coords_centered"]
                
                propability_ = np.copy(propability)
                ph_l_shape_half = np.divide(phantom_label.shape, 2)
                even = np.logical_not(ph_l_shape_half % 2)
                invalid_s = np.floor(ph_l_shape_half).astype(int)
                invalid_e = invalid_s - even
                propability_[0:invalid_s[0], ...] = 0; propability_[-invalid_e[0]:, ...] = 0
                propability_[:, 0:invalid_s[1], :] = 0; propability_[:, -invalid_e[1]:, :] = 0
                propability_[..., 0:invalid_s[2]] = 0; propability_[..., -invalid_e[2]:] = 0
                
                coords_centered = props["coords_centered"]

                pos_tried = 0
                s_pos = time.time()
                while pos_tried < self.break_criterion_positions:
                    counter_pos_tries += 1
                    pos_tried += 1
                    # random position to place the object
                    prop_sum = np.sum(propability_)
                    if prop_sum == 0:
                        break
                    elif prop_sum != 1:
                        propability_ /= prop_sum
                    pos_ind = np.random.choice(range(c.shape[0]), p=propability_[tuple(c.T)])
                    propability_[tuple(c[pos_ind,:].T)] = 0
                    value = c[pos_ind,:] + bb[0:ndim]
                    coords = np.add(coords_centered, value)

                    # calculate overlap with the given mask (100% of phantom should be inside for later brightness reduction)
                    overlap_with_mask = np.nonzero(mask[coords[:, 0], coords[:, 1], coords[:, 2]]==cell_prop.label)[0]
                    overlap_count = len(overlap_with_mask)

                    if overlap_count >= props["area"]*0.95:
                        valid_coords = coords[overlap_with_mask,:]
                        valid_label_coords = props["label_coords"][overlap_with_mask,:]
                        # Place phantom
                        label_sim[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = self.phantom_num
                        image_sim[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = phantom_image[
                                                                                valid_label_coords[:, 0],
                                                                                valid_label_coords[:, 1],
                                                                                valid_label_coords[:, 2]
                                                                            ]
                        phantom_types.append(folder_ind_list[nuclei_nr])
                        self.phantom_num += 1
                        objects_tried = 0
                        object_placed = 1
                        nuclei_nr_amount[nuclei_nr] += 1
                        # print('Object_Placed!')
                        break # -> next object
                d_pos = time.time() - s_pos; dc_pos += d_pos

                if not (objects_tried < self.break_criterion_objects):
                    counter_aborted += 1
        
        return image_sim, label_sim, phantom_types

    def get_morph_props(self, props, px_size):
        stats = {
            'eccentricity': [],
            'orientation_xy': [],
            'volume_um': [],
        }

        for prop in props:
            coords = prop.coords * (self.px_size_sim_img[0]/self.px_size_sim_img[1], 1, 1)
            cov = np.cov(coords, rowvar=False)
            e_w, e_v = np.linalg.eig(cov) # e_v[:,i]
            ind_e_s = np.flip(np.argsort(e_w))
            # Roundness based on the length difference of major and minor axis length
            stats["eccentricity"].append(np.sqrt(1-(e_w[ind_e_s[-1]] / e_w[ind_e_s[0]])))
            stats["orientation_xy"].append(np.arctan(e_v[:, ind_e_s[0]][1] / e_v[:, ind_e_s[0]][2]))
            stats["volume_um"].append(prop.area * np.prod(px_size))
        return stats


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
        

        ### Mask output
        if verbose:
            print("Saving cell labels ...")
            s_PIG_cell_labels = time.time()
        label_mask = load_data(self.path_spheroid_mask)[:-self.z_append]
        tiff.imwrite(os.path.join(self.labels_folder, "{:03d}_{:03d}_final_cell_labels.tif".format(mask_num, img_num)), label_mask)
        del label_mask
        if verbose == 2:
            d_PIG_cell_labels = time.time() - s_PIG_cell_labels; print(f"Saving cell labels required {d_PIG_cell_labels:.1f} s -> {d_PIG_cell_labels/60:.1f} min")

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