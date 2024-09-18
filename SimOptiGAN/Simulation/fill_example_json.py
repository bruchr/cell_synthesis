import json


if __name__ == "__main__":
    params = dict()

    # path that the resulting images are saved to
    params["output_path"] = "./out"
    
    # files to extract the cells/phantoms from
    params["input_phantom_folders"] = ["./images/extracted_phantoms"]
    params["input_phantom_folders_prop"] = [1]
    
    # path to binary images, which define the possible positions for nuclei placement
    params["input_spheroid_mask"] = "./images/extracted_masks/spheroid_mask_0.tif"  # "hemisphere.tif"
    # paths to the point spread functions (in order of usage from front to back)
    params["paths_psf"] = [
        "./images/extraxted_psf_zoom1d16_32b.tif",
        "./images/extraxted32b_pntMan_psf_zoom1d0_combined.tif"
    ]
    
    # use gpu for convolution
    params["use_gpu_conv"] = 1
    # shape of the image that's generated (before synthesizing the downsampling of the camera!)
    # can be left empty () to use the mask shapes
    params["generated_image_shape"] = (77, 1024, 1024)  # (50, 1024, 1024)
    # how many images are to be generated
    params["num_of_images"] = 1
    
    
    params["max_overlap"] = 0 # max overlap of the cells in the phantom image
    params["min_dist"] = 0 # minimal distance between two nuclei
    params["break_criterion_positions"] = 50 # max no. of consecutive failed placement. Step 1: no. of drawn positions
    params["break_criterion_objects"] = 50 # max no. of consecutive failed placement. Step 2: no. of drawn objects
    
    params["brightness_red_fuction"] = "f3p"
    params["brightness_red_factor_b"] = 200

    params["exposure_percentage"] = 1.0 # exposure percentage. >1 is overexposed 
    params["dc_baseline"] = 15  # baseline as a simplified dark current (in range [0, 255])
    params["noise_gauss"] = 1.7  # sigma for the gauss noise (in range [0, 255])
    params["noise_gauss_absolute"] = 0  # sigma for the absolute gauss noise (in range [0, 255])
    params["noise_poisson_factor"] = 0 # scale of the poisson noise
    params["num_of_acquisition_iterations"] = 1 # no. of images used for averaging

    # pixel size of the original and therefore the generated image (before the camera effects) and desired pixel size
    params["px_size_phantom_img"] = [0.4501347, 0.0610951, 0.0610951] # px size of the nuclei phantoms
    params["px_size_sim_img"] = [1.5001311, 0.488759, 0.488759] # px size of the generated image before downsampling
    params["px_size_desired"] = [1.5001311, 0.488759, 0.488759] # desired px size of output image

    # fragments
    params["input_fragment_folder"] = "./images/extracted_fragments"
    params["fragment_probability"] = 0.2

    # subcultures
    params["input_subculture_folder"] = "./images/extracted_subculture_phantoms"
    params["num_of_subcultures"] = 0
    params["subculture_radius_xy"] = 175
    params["subculture_radius_z"] = 10
    params["subculture_min_dist"] = 0
    params["px_size_subculture_phantoms"] = [0.3250898, 0.04253006666, 0.04253006666]

    f = open("params.json", "w")
    f.write(json.dumps(params, indent=4))
    f.close()
