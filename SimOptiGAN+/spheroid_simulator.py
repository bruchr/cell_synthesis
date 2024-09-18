import json
import os.path
from typing import List, Optional

import numpy as np
from skimage.measure import regionprops
import tifffile as tiff

from modules.camera_acquisition_simulation import CameraAcquisitionSimulation
from modules.optical_system_simulation import OpticalSystemSimulation
from modules.phantom_image_generator import PhantomImageGenerator
from utils import prepare_and_save_mask, delete_mask, load_data


class SpheroidSimulator:
    phantom_image_generator: PhantomImageGenerator
    optical_system_simulation: OpticalSystemSimulation
    camera_acquisition_simulation: CameraAcquisitionSimulation

    path_spheroid_mask_list: List[str]
    num_of_images: int
    path_output: str
    path_out_folders: List[str]
    generation_params: dict
    optical_system_params: dict
    camera_acquisition_params: dict

    params_set: bool

    def __init__(self) -> None:
        self.phantom_image_generator = PhantomImageGenerator()
        self.optical_system_simulation = OpticalSystemSimulation()
        self.camera_acquisition_simulation = CameraAcquisitionSimulation()

        self.path_out_folders = ["images", "labels"]
        self.params_set = False

    @staticmethod
    def missing_param(params: dict) -> Optional[str]:
        """
        Checks if all necessary params are given and returns None.
        If params are missing, the key/name of the first one is returned.

        Args:
            params: A dict containing all parameters for the simulation process

        Returns:
            None if no key is missing, otherwise the key/name (str) of the first missing parameter
        """
        necessary_keys = [
            "output_path", "input_phantom_folders", "input_phantom_folders_prop", "input_spheroid_mask", "paths_psf", "use_gpu_conv", "generated_image_shape", 
            "num_of_images", "volume_ratio_phantom", "max_overlap", "break_criterion_positions", "break_criterion_objects",
            "brightness_red_fuction", "brightness_red_factor_b",
            "dc_baseline", "noise_gauss", "noise_gauss_absolute", "noise_poisson_factor", "num_of_acquisition_iterations", 
            "px_size_phantom_img", "px_size_mask_img", "px_size_sim_img", "px_size_desired"
            ]

        for key in necessary_keys:
            if key not in params:
                return key

        return None

    def import_settings(self, params: dict) -> None:
        """
        Imports the given settings for the simulation process.

        Args:
            params: A dict containing all parameters for the simulation process

        Returns:
            None
        """
        missing = self.missing_param(params)
        if missing is not None:
            raise KeyError("Key '" + missing + "' is missing in the given parameters")

        self.path_spheroid_mask_list = params["input_spheroid_mask"]
        self.num_of_images = params["num_of_images"]
        self.path_output = params["output_path"]
        self.path_spheroid_mask_tmp = os.path.join(self.path_output, self.path_out_folders[0], 'tmp_spheroid_mask.tif')

        self.z_append = round(30 / (params["px_size_sim_img"][0]/params["px_size_phantom_img"][0]))
        if len(params["generated_image_shape"]) != 0:
            self.shapes_des = [np.asarray(params["generated_image_shape"]) for _ in range(len(params["input_spheroid_mask"]))]
        else:
            # Keep correct px size of mask equal to that of the simulation
            self.shapes_des = []
            for path_spher_mask in params["input_spheroid_mask"]:
                self.shapes_des.append(np.asarray(tiff.imread(path_spher_mask).shape) * np.divide(params["px_size_mask_img"], params["px_size_desired"]))
        self.shapes_sim = []
        for shape_des in self.shapes_des:
            if not np.array_equal(params["px_size_sim_img"], params["px_size_desired"]):
                shape_sim = np.divide(shape_des, np.divide(params["px_size_sim_img"], params["px_size_desired"]))
            else:
                shape_sim = shape_des
            shape_sim[0] += self.z_append
            self.shapes_sim.append(np.round(shape_sim).astype(int))

        self.generation_params = {
            "path_phantom_folders": params["input_phantom_folders"],
            "phantom_folders_prop": params["input_phantom_folders_prop"],
            "path_spheroid_mask": None,
            "shape": None,
            "z_append": self.z_append,
            "volume_ratio_phantom": params["volume_ratio_phantom"],
            "max_overlap": params["max_overlap"],
            "break_criterion_positions": params["break_criterion_positions"],
            "break_criterion_objects": params["break_criterion_objects"],
            "px_size_phantoms": params["px_size_phantom_img"],
            "px_size_mask": params["px_size_mask_img"],
            "px_size_sim_img": params["px_size_sim_img"],
            "images_folder": os.path.join(self.path_output, self.path_out_folders[0]),
            "labels_folder": os.path.join(self.path_output, self.path_out_folders[1])
        }

        self.optical_system_params = {
            "paths_psf": params["paths_psf"],
            "path_spheroid_mask": None,
            "gpu_conv": params["use_gpu_conv"],
            "z_append": self.z_append,
            "shape_des": None,
            "brightness_red_fuction": params["brightness_red_fuction"],
            "brightness_red_factor": params["brightness_red_factor_b"],
            "path_output": os.path.join(self.path_output, self.path_out_folders[0])
        }

        self.camera_acquisition_params = {
            "exposure_percentage": params["exposure_percentage"],
            "baseline": params["dc_baseline"],
            "noise_gauss": params["noise_gauss"],
            "noise_gauss_absolute": params["noise_gauss_absolute"],
            "noise_poisson_factor": params["noise_poisson_factor"],
            "num_of_acquisition_iterations": params["num_of_acquisition_iterations"],
            "px_size_img": params["px_size_sim_img"],
            "px_size_desired": params["px_size_desired"],
            "path_output": os.path.join(self.path_output, self.path_out_folders[0])
        }

        self.params_set = True

    def import_settings_json(self, file_path: str) -> None:
        """
        Tries to open and import parameters from a given text file in java object notation

        Args:
            file_path: path to the text file

        Returns:
            None
        """
        f = open(file_path, "r")
        params = json.loads(f.read())
        f.close()

        self.import_settings(params)

    def check_n_make_out_folders(self) -> None:
        """
        Checks if all necessary output folders are available. If not, they are created.

        Returns:

        """
        for folder in self.path_out_folders:
            path_out = os.path.join(self.path_output, folder)
            if not os.path.isdir(path_out):
                os.makedirs(path_out)

    def run(self, verbose: bool = False, save_interim_images: bool = False) -> None:
        """
        Runs the simulation process with the given parameters.

        Args:
            verbose: if verbose is set, the command line output is more verbose
            save_interim_images: if set, all interim results (e.g. the phantom image) are saved

        Returns:
            None
        """
        if not self.params_set:
            raise AttributeError("No parameters set. Call import_settings or import_settings_json first!")

        self.check_n_make_out_folders()

        for n_mask, path_spher_mask in enumerate(self.path_spheroid_mask_list):
            self.generation_params['shape'] = self.shapes_sim[n_mask]
            self.generation_params['path_spheroid_mask'] = self.path_spheroid_mask_tmp # path_spher_mask
            self.optical_system_params['path_spheroid_mask'] = self.path_spheroid_mask_tmp # path_spher_mask
            self.optical_system_params['shape_des'] = self.shapes_des[n_mask]

            self.phantom_image_generator.import_settings(self.generation_params)
            self.optical_system_simulation.import_settings(self.optical_system_params)
            self.camera_acquisition_simulation.import_settings(self.camera_acquisition_params)

            for n in range(1, self.num_of_images + 1):
                print("Generating image {}_{}".format(n_mask, n))
                if verbose:
                    print(f"\nDesired image shape:           {self.shapes_des[n_mask][0]:4.0f} x {self.shapes_des[n_mask][1]:4.0f} x {self.shapes_des[n_mask][2]:4.0f} px^3")
                    print(f"Image shape during simulation: {self.shapes_sim[n_mask][0]:4.0f} x {self.shapes_sim[n_mask][1]:4.0f} x {self.shapes_sim[n_mask][2]:4.0f} px^3")
                    req_mem = (np.prod(self.shapes_sim[n_mask], dtype=np.float32) * 4 * 2) * 1e-9 # GB
                    print(f"Predicted peak memory usage: {req_mem:.1f} GB")

                prepare_and_save_mask(path_spher_mask, self.path_spheroid_mask_tmp, self.z_append, self.shapes_sim[n_mask])
                
                os.path.join(self.path_output, self.path_out_folders[1], f'{n_mask:03d}_000_final_cell_labels.tif')

                ### phantom_image_generator
                image, label, phantom_types = self.phantom_image_generator.run(mask_num=n_mask, img_num=n, verbose=verbose, save_interim_images=save_interim_images)
                # Reducing ram usage by temporarily deleting label image, as it is not required for the optical system simulation.
                label = label[:-self.generation_params['z_append'], :, :]
                tiff.imwrite(os.path.join(self.path_output, self.path_out_folders[1], "{:03d}_{:03d}_final_label.tif".format(n_mask, n)), label)
                del label
                

                ### optical_system_simulation
                image = self.optical_system_simulation.run(image, mask_num=n_mask, img_num=n, verbose=verbose, save_interim_images=save_interim_images)
                image = image[:-self.generation_params['z_append'], :, :]


                ### camera_acquisition_simulation
                label = tiff.imread(os.path.join(self.path_output, self.path_out_folders[1], "{:03d}_{:03d}_final_label.tif".format(n_mask, n)))
                image, label = self.camera_acquisition_simulation.run(image, label, mask_num=n_mask, img_num=n, verbose=verbose,
                                                                      save_interim_images=save_interim_images)

                tiff.imwrite(os.path.join(self.path_output, self.path_out_folders[0], "{:03d}_{:03d}_final.tif".format(n_mask, n)), image)
                tiff.imwrite(os.path.join(self.path_output, self.path_out_folders[1], "{:03d}_{:03d}_final_label.tif".format(n_mask, n)), label)

                
                del image
                label_type = np.zeros_like(label, dtype=np.uint16)
                label_props = regionprops(label)
                for label_prop in label_props:
                    # -1: Fragments, 0:Folder0, 1:Folder1
                    phantom_type = phantom_types[label_prop.label-1] +1
                    label_type[tuple(label_prop.coords.T)] = phantom_type
                
                tiff.imwrite(os.path.join(self.path_output, self.path_out_folders[1], "{:03d}_{:03d}_final_label_type.tif".format(n_mask, n)),
                            label_type)

                if verbose:
                    print("\nSaved final image")
                    print("-"*60)
            
            delete_mask(self.path_spheroid_mask_tmp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Generate synthetic microscopy images of cell spheroids")
    parser.add_argument("-json", type=str, default="./params.json",
                        help="The json-file, that contains all relevant parameters")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False, required=False,
                        help="If verbose is set, the command line output is more detailed")
    parser.add_argument("-vt", "--verbose_time", dest="verbose_time", action="store_true", default=False, required=False,
                        help="If verbose_time is set, the command line output will show the required times for individual steps")
    parser.add_argument("--save_interim", dest="save_interim", action="store_true", default=False, required=False,
                        help="If save_interim is set, interim results are saved (e.g. phantom image)")
    args = parser.parse_args()

    verbose_arg = vars(args)["verbose"] if not vars(args)["verbose_time"] else 2
    save_interim_arg = vars(args)["save_interim"]
    json_file = vars(args)["json"]
    if not os.path.isfile(json_file):
        if json_file != parser.get_default("json"):
            error_message = "The given json '" + json_file + "' is no valid file"
        else:
            error_message = "Please specify an existing file for the parameters using the command line argument '-json'"
        raise AttributeError(error_message)

    simulator = SpheroidSimulator()
    simulator.import_settings_json(json_file)
    simulator.run(verbose=verbose_arg, save_interim_images=save_interim_arg)
