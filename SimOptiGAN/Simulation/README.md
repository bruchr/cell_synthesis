SpheroidSimulator
=================

This package provides a pipeline to generate synthetic microscopy images of three dimensional cell cultures.

### Usage
The python script spheroid_simulator can be started using the command line.
There are three optional command line arguments:
* *-json <path>*: The path to a file in java object notation including all parameters necessary to use the pipeline.
If json is not set, the script tries to use a json file in the working directory.
* *--verbose*: If this argument is set, the output on the command line is more verbose
* *--save_interim*: If this argument is set, the pipeline saves all interim results like the phantom image or the mask used to reduce the image intensity