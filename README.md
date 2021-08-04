
## Requirements
 * PyTorch 1.6.0
 * CUDA 10.1 (to allow use of GPU, not compulsory)

## Running the Software

All the python classes and functions strictly needed to implement the USAD architecture can be found in `usad.py`.
An example of an application deployed with the [SWaT dataset] is included in `USAD.ipynb`.


## Publication

Audibert et al. [USAD : UnSupervised Anomaly Detection on multivariate time series]. 2020

[SWaT dataset]: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat
[USAD : UnSupervised Anomaly Detection on multivariate time series]: https://dl.acm.org/doi/pdf/10.1145/3394486.3403392


## anomaly 判斷
* 這邊就是windows裡面只要有一個是anomaly整段windows都標記成anomaly

## usage
>>  it can run on 國網(gpu mem 大小的問題)
>>  you also can reduce batch size to prevent it from lack of mem

* put data in input/
* modify paramters on the top of main.py
* help: python main.py

## result
* result is ROC.png and history.png


## modify
if you want to add new model:
1. add new model in model.py 
2. add "if else condition" in main.py exection.test() and exection.train()

## docker
https://hub.docker.com/repository/docker/djjjimmyyy/anomaly_detection