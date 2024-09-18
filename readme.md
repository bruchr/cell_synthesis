# Improving 3D deep learning segmentation with biophysically motivated cell synthesis
**Roman Bruch, Mario Vitacolonna, Elina Nürnberg, Simeon Sauer, Rüdiger Rudolf and Markus Reischl**

## Installation

Clone this repository using [`git`](https://git-scm.com/downloads) or download it as `.zip`file and extract it.

Install a conda distribution like [Anaconda](https://www.anaconda.com/products/individual).

Create the environment with conda:
```
conda env create -f environment.yml
```

Activate the environment:
```
conda activate cell_syn
```
Once the environment is activated, the submodules can be run as described below.


## Data

The example data to run this code can be found at Zenodo:
<https://doi.org/10.5281/zenodo.11240362>

Extract the folder and move it to the same directory as this repository. The folder structure should look like this:
```
├── cell_data_synthesis
│   ├── Data
│   │   ...
│   ├── Mem2NucGAN-P
│   │   ...
│   ...
...
```

## SimOptiGAN

### Simulation

The original, but outdated code of SimOptiGAN can be found at [`cell_data_synthesis`](https://github.com/bruchr/cell_data_synthesis). This repository includes the updated version of SimOptiGAN.

A `.json` file is required for parameterization of the pipeline. One can either copy `./Simulation/params_template.json` and adapt it, or generate a new one with ```python .SimOptiGAN/Simulation/fill_example_json.py```. The settings file used in the paper can be found at `./Data/SimOptiGAN/normal_res_run_v6/params.json`.

The simulation is started via the command line:
```
python ./SimOptiGAN/Simulation/spheroid_simulator.py -json ./SimOptiGAN/Simulation/params.json
```

Optionally the parameters `--verbose` (`-v`), `--verbose_time` (`-vt`) and `--save_interim` can be used. The verbose parameter provides a more verbose output. The verbose time parameter provides additional runtime measurements of individual steps. If *save_interim* is given, intermediate image results are saved in the specified output directory.

Once the simulation is completed, the generated images can be optimized. For this, a dataset with real and simulated images needs to be created with the following folder structure:
```
├── name_of_dataset
│   ├── trainA
│   │   ...
│   ├── trainB
│   │   ...
│   ├── inferenceA
│   │   ...
│   ├── inferenceB
│   │   ...
...
```
The structure and naming is required for the files to be correctly detected. The folders `inference#` are not required for model training, but only for model inference.
The training dataset used in the paper can be found at `./Data/Optimization_Model/Train_Data/normal_res_td_v6`.


### Optimization Training

Insert the real images in `trainA` and the simulated images in `trainB`. Copy the template settings file `./SimOptiGAN/Optimization/settings_template.hjson`, rename it to `settings.hjson` and adjust the parameters. Set an experiment name, insert the dataset path at *dataset_folder* and set the *mode* to 'train'. The parameters *direction* and *direction_inference* should be 'BtoA' if the real images are located in *trainA*. Otherwise the settings file as used in the paper can be utilized: `./Data/Optimization_Model/Train_Data/settings_ht29_td6_normal_res.hjson`.

**Note**: Some parameters specified in the settings file can be overwritten with command line options. See 
```python ./SimOptiGAN/Optimization/start.py --help``` for more details.

Start the training of the optimization with:
```
python ./SimOptiGAN/Optimization/start.py --settings ./SimOptiGAN/Optimization/settings.hjson
```

**Note**: the GPU memory requirements for training the 3D CycleGAN are quite high. If memory issues occur, reduce batch size or image/crop size to (32, 128, 128). Adjust *inf_patch_size* according to the training image/crop size for optimal results.


### Optimization Inference

Once the training is completed, the network can be used to optimize the simulated data. Insert simulated images in `inferenceB` located in the train dataset folder. The trained models can be found at the specified output folder. Note: the experiment name is appended by a timestamp and the epoch after which the modes were saved. Use the settings file located in the same folder as the desired model for inference.

The trained model and corresponding settings as used in the paper can be found at `./Data/Optimization_Model/ht29_td6_normal_res_run1_epoch_2994`.

The inference of the model used in the paper can be started with:
```
python ./SimOptiGAN/Optimization/start.py --settings ./Data/Optimization_Model/ht29_td6_normal_res_run1_epoch_2994/settings.hjson --mode inference
```
This command transforms the images located in `inferenceB` of dataset *normal_res_td_v6*. Results will be placed in the model's folder.

## SimOptiGAN+

The process for SimOptiGAN+ is similar to that of SimOptiGAN. Therefore only the differences are described.

The settings file for the simulation as used in the paper can be found at `./Data/SimOptiGAN+/3D_NucPlacement_V5d1/params.json`.
The simulation is started via the command line:
```
python ./SimOptiGAN+/spheroid_simulator.py -json ./SimOptiGAN+/params.json
```

In the paper, the same optimization model trained for SimOptiGAN was used for SimOptiGAN+.


## Mem2NucGAN-P

For the training of Mem2NucGAN-P, paired images of real nuclei signals and corresponding segmented membranes, along with unpaired segmented nuclei signals are required. The training of the pix2pix based model requires the following folder structure:
```
├── name_of_dataset
│   ├── trainA
│   │   ...
│   ├── trainB
│   │   ...
│   ├── trainSeg
│   │   ...
│   ├── inferenceA
│   │   ...
│   ├── inferenceB
│   │   ...
...
```
The structure and naming is required for the files to be correctly detected. Folders `trainA`, `trainB`, and `trainSeg` should contain membrane segmentations, corresponding real nuclei signals, and arbitrary nuclei segmentations, respectively. The folders `inference#` are only required for model inference.

The training procedure is similar to that of the optimization model of SimOptiGAN. Please refer to the corresponding description for more information.

The dataset used in the paper can be found at `./Data/Mem2NucGAN-P/Train_Data/segMembrane2nucSeg_td1`.
The settings file utilized in the paper is located at `./Data/Mem2NucGAN-P/Train_Data/settings.hjson`.

Training of the transformation model can be started with:
```
python ./Mem2NucGAN-P/start.py --settings ./Data/Mem2NucGAN-P/Train_Data/settings.hjson
```

Inference of the model used in the paper can be started with:
```
python ./Mem2NucGAN-P/start.py --settings ./Data/Mem2NucGAN-P/segMembrane2nucSeg_td1_run1_epoch_2800/settings.hjson --mode inference
```


## Mem2NucGAN-U

For the training of Mem2NucGAN-U, unpaired images of real nuclei and membrane signals, along with arbitrary segmented nuclei signals, are required. The training of the CycleGAN based model requires the following folder structure.
```
├── name_of_dataset
│   ├── trainA
│   │   ...
│   ├── trainB
│   │   ...
│   ├── trainSeg
│   │   ...
│   ├── inferenceA
│   │   ...
│   ├── inferenceB
│   │   ...
...
```
The folders `trainA`, `trainB` and `trainSeg` should contain synthetic membrane signals, real nuclei signals and arbitrary nuclei segmentations, respectively.

The training procedure is similar to that of the optimization model of SimOptiGAN. Please refer to the corresponding description for more information.

The dataset used in the paper can be found at `./Data/Mem2NucGAN-U/Train_Data/settings.hjson`.
The settings file utilized in the paper is located at `./Data/Mem2NucGAN-U/Train_Data/settings.hjson`.

Training of the transformation model can be started with:
```
python ./Mem2NucGAN-U/start.py --settings ./Data/Mem2NucGAN-U/Train_Data/settings.hjson
```

Inference of the model used in the paper can be started with:
```
python ./Mem2NucGAN-U/start.py --settings ./Data/Mem2NucGAN-U/binMembrane2nucSeg_td1_run4_epoch_2800/settings.hjson --mode inference
```


## Segmentation

For the segmentation based evaluation we used the StarDist model ([StarDist GitHub](https://github.com/stardist/stardist)). Please refer to this repository for instructions.
For training and inference of StarDist models, the example script files were adapted. The adapted versions are located at `./Segmentation/2_training.py`.