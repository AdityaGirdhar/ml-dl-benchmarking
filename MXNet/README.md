# MXNet
In part of the study to compare between the libraries, MXNet is used as a popular Deep Learning library, with possible GPU acceleration to run advanced models. The goal of this study is to compare between the different libraries, and to see how they perform on different tasks. 

# Table of Contents
1. [Installation and Setup](#installation-and-setup)


# Installation and Setup 
In order to setup and run the code of MXNet on your machine, you need to follow the following steps: 
1. Install Anaconda according to your OS from [here](https://www.anaconda.com/download/). 
2. Check whether you have an NVIDIA GPU on your system or not. If you have an NVIDIA GPU, you need to check whether your GPU is CUDA enabled or not. To check, please [click here](https://developer.nvidia.com/cuda-gpus).
3. If you have a CUDA enabled GPU, you need to install CUDA Toolkit of version CUDA 9.2 from [here](https://developer.nvidia.com/cuda-92-download-archive)
4. Now, you need to install cuDNN of a version that is compatible with the CUDA Toolkit version that you have installed. To do so, you need to register to the NVIDIA Developer Program from [here](https://developer.nvidia.com/cudnn). After registering, you can download the cuDNN library from [here](https://developer.nvidia.com/rdp/cudnn-download).

After this, you can directly create a new environment by cloning the YML file. To do so, follow from [Alternative step 5](#alternative-step-5).

5. Now, you need to create a conda environment with the following command: 
```
conda create -n myenv python=3.6
```
6. Once created, activate the environment with the following command: 
```
conda activate myenv
```
7. Now, you need to install MXNet with the following command: 
```
pip install mxnet-cu92
```
8. Now, you need to install Jupyter Notebook with the following command: 
```
pip install jupyter
```
9. Now, you need to install the following libraries: (although numpy will be installed with MXNet)
```
pip install numpy
pip install pandas
```
10. To check whether you have correctly installed a library X, simply run the following command: 
```
conda list X
```
11. To run the Jupyter Notebook, you need to run the following command: 
```
jupyter notebook
```

### Alternative step 5
5. Now, you need to create a conda environment with the following command: 
```
conda env create -f dependencies.yml
```
Use the YML file provided in this repository in- [dependencies.yml](Dependencies/requirements_GPU.yml).
After running the above command, your environment will be created with all the dependencies installed.
6. To activate the environment, run the following command: 
```
conda activate mxnet
```
7. To run the Jupyter Notebook, you need to run the following command: 
```
jupyter notebook
```