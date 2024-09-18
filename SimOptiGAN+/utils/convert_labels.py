import os

import tifffile as tiff

def normal_to_sparse(path_label_folder:str, path_out:str=None)->None:
    '''Converts normal labels to sparse labels. Sparse labels have an additional class 0 or ignore, for not labeled regions.
    If the output path is not given, a new folder labels is created in the parent folder.'''
    
    for ind, f_name in enumerate(os.listdir(path_label_folder)):
        if f_name.endswith('.tif'):
            f_path = os.path.join(path_label_folder, f_name)
            label_sparse = tiff.imread(f_path) + 1
            if path_out is None:
                out_path = os.path.dirname(os.path.normpath(path_label_folder))
                out_path = os.path.join(out_path, 'labels_sparse', f_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=ind)
            tiff.imwrite(out_path, label_sparse)

def sparse_to_normal(path_label_folder:str, path_out:str=None)->None:
    '''Converts sparse labels to normal labels. Sparse labels have an additional class 0 or ignore, for not labeled regions.
    If the output path is not given, a new folder labels is created in the parent folder.
    The class 0 or ignore is converted to background.'''
    
    for ind, f_name in enumerate(os.listdir(path_label_folder)):
        if f_name.endswith('.tif'):
            f_path = os.path.join(path_label_folder, f_name)
            label_sparse = tiff.imread(f_path)
            label_sparse[label_sparse==0] = 1
            label = label_sparse - 1
            if path_out is None:
                out_path = os.path.dirname(os.path.normpath(path_label_folder))
                out_path = os.path.join(out_path, 'labels', f_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=ind)
            tiff.imwrite(out_path, label)



if __name__ == "__main__":
    
    folder_sparse = "C:/Users/xx3662/Desktop/Projekte_Daten/Data_Synthesis/Simulation/Runs/Mario_A549_Mono_96h_Doxo_0uM_Ki/Run_4_worse_quality/labels_sparse"
    sparse_to_normal(folder_sparse)