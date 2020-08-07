# USAD

Scripts and utility programs for implementing the USAD architecture

Author: Francesco Galati

Additional contributions: Maria A Zuluaga, Julien Audibert


# How to cite


If you use this software, please cite the following paper as appropriate:

    Audibert, J., Michiardi, P., Guyard, F., Marti, S., Zuluaga, M. A. (2020).
    USAD : UnSupervised Anomaly Detection on multivariate time series.
    Physics in Medicine & Biology, 59(9), 2155.??????????????????????

## Citations for cardiac-tools
    Maria A. Zuluaga, M. Jorge Cardoso, Marc Modat, Sébastien Ourselin.
    Multi-atlas Propagation Whole Heart Segmentation from MRI and CTA Using a Local Normalised Correlation
    Coefficient Criterion.
    Functional Imaging and Modeling of the Heart, vol 7945, Lecture Notes in Computer Science, 174-181


# Building the software

## Requirements
 * CMake (3 or higher)
 * ITK 4.8 (not tested with later versions)
 * Python (to allow use of pip to install imagesplit)
 * ImageSplit (`pip install imagesplit`)
 * Fiji
 * You must then build the C++ components as described below


## Build instructions for the C++ components

The C++ components of the code should be built using CMake.
Use ccmake to configure the project. Make a build directory in a different location to your source code and run the following, or else use `cmake-gui`.

```
    mkdir vessel-tools-build
    cd vessel-tools-build
    ccmake <path-to-vessel-tools>
```

Then press `c` to configure. If you see errors, you may need to set variables (such as for your ITK location).
Once the configuration is complete, you can generate the build files by pressing 'g'.
Then you can build the project using `make`:

```
    make -j
```


## Installing the python ImageSplit package

The python package ImageSplit is used in some of the bash scripts. If you need it, the easiest way is to install using `pip` (assuming you have Python installed).
Note: if you are using the system default python (especially on MacOS), it is recommended that you do not modify the system installation. Instead, either install ImageSplit locally, or install an alternative version of Python, or use `virtualenv` to create your own python virtual environment.

# Running the Software

The make process will create a number of executables which can be run from the command-line.
Please refer to the codebase or help output of the commands for details of the options.

There are also a number of bash scripts and Fiji scripts. You will need to modify these appropriately to suit your data.


## Example script: placental analysis

An example placental analysis bash script is included here: `scripts/analyse_placenta.sh`
This is based on analysing a microCT volume, but other data types can be used with appropriate modification.
First you must install and compile the required software as described above, and configure the script for your paths.



# Copyright and licensing

Copyright 2018 University College London.

vessel-tools is released under the BSD-3 license. Please see the `license file`_ for details.


Acknowledgements
----------------

This work was supported through an Innovative Engineering for Health award by the [Wellcome Trust][wellcometrust] [WT101957], the [Engineering and Physical Sciences Research Council (EPSRC)][epsrc] [NS/A000027/1] and a [National Institute for Health Research][nihr] Biomedical Research Centre [UCLH][uclh]/UCL High Impact Initiative.



[tig]: http://cmictig.cs.ucl.ac.uk
[giftsurg]: http://www.gift-surg.ac.uk
[cmic]: http://cmic.cs.ucl.ac.uk
[ucl]: http://www.ucl.ac.uk
[nihr]: http://www.nihr.ac.uk/research
[uclh]: http://www.uclh.nhs.uk
[epsrc]: http://www.epsrc.ac.uk
[wellcometrust]: http://www.wellcome.ac.uk
[githubhome]: https://github.com/gift-surg/vessel-tools
