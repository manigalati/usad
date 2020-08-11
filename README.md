# USAD - UnSupervised Anomaly Detection on multivariate time series

Scripts and utility programs for implementing the USAD architecture.

Implementation by: Francesco Galati.

Additional contributions: Julien Audibert, Maria A. Zuluaga.

## How to cite

If you use this software, please cite the following paper as appropriate:

    Audibert, J., Michiardi, P., Guyard, F., Marti, S., Zuluaga, M. A. (2020).
    USAD : UnSupervised Anomaly Detection on multivariate time series.
    26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, August 23-27, 2020, San Diego, USA

## Requirements
 * PyTorch 1.6.0
 * CUDA 10.1 (to allow use of GPU, not compulsory)

## Running the Software

All the python classes and functions strictly needed to implement the USAD architecture can be found in `usad.py`.
An example of an application deployed with the [SWaT dataset] is included in `USAD.ipynb`.

## Copyright and licensing

Copyright 2020 Eurecom.

This software is released under the BSD-3 license. Please see the license file_ for details.

[SWaT dataset]: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat
