# USAD

Scripts and utility programs for implementing the USAD architecture

Author: Francesco Galati

Additional contributions: Maria A Zuluaga, Julien Audibert

# How to cite

If you use this software, please cite the following paper as appropriate:

    Audibert, J., Michiardi, P., Guyard, F., Marti, S., Zuluaga, M. A. (2020).
    USAD : UnSupervised Anomaly Detection on multivariate time series.
    Physics in Medicine & Biology, 59(9), 2155.??????????????????????

# Building the software

## Requirements
 * CMake (3 or higher)
 * ITK 4.8 (not tested with later versions)
 * Python (to allow use of pip to install imagesplit)
 * ImageSplit (`pip install imagesplit`)
 * Fiji
 * You must then build the C++ components as described below

# Running the Software

All the python classes and functions strictly needed to implement the USAD architecture can be found in `usad.py`.
An example of a realistic application deployed with the [SWaT dataset] is included in `USAD.ipynb`.

# Copyright and licensing

Copyright 2020 Eurecom ????????????.

[SWaT dataset]: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat
