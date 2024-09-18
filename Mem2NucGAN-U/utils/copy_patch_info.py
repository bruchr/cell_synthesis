import os
from shutil import copyfile


def copy_patch_info(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            f_path_in = os.path.join(input_folder, file_name)
            f_path_out = os.path.join(output_folder, file_name)
            copyfile(f_path_in, f_path_out)