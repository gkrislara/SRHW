# Super Resolution in Zynq Ultrascale+ MPSoC - Iteration 2
* * *

For iteration 1 refer [Edge Inference of Image Super Resolution Deep Learning Models](https://github.com/gkrislara/Image-super-resolution-FPGA)

## Tools

* Vitis 2020.2 (includes Vivado and Vivado HLS)
* Vitis AI v1.3
* Petalinux 2020.2
* Docker

## Requirements

* Development Environment with RAM: 32GB Disk Space: 500GB or higher
* Nvidia GPU with compute capability > 6.0 with sufficient memory (>4GB) - GeForce 1050 Ti is used in this case
* HD and UHD Monitors ( Acer KA is used) and BARCO ( for displaying results)
* Zynq Ultrascale board ( ZCU102 is used)
* Ethernet Cables for connectivity between systems (can be used to display on host system if there is no monitor available)

## Prerequisites
* Knowledge on FPGA and Integrating IPs in Vivado
* Knowledge on Computer Architecture and Embedded Systems
* Linux Operating system - Petalinux/Yocto
* Experience in building Deep learning models
* Programming Languages: Cpp - 17 and Python - 3.8+
* Frameworks/libraries: Opencv,Numpy,Tensorflow and Pytorch

This documentation describes the state the repo and is divided into four sections

1. Training
2. Quantization
3. Hardware Development
4. Application Development

For complete information about the experiments and iterations follow [here](https://htic.atlassian.net/wiki/spaces/~11214967/pages/93487268/Daily+Work+Update) from 29/10/2020 to 15/01/2021

For How-To's and project flows and other features follow [Vitis AI User Documentation](https://www.xilinx.com/html_docs/vitis_ai/1_3/index.html) and [Vitis-AI Github](https://github.com/Xilinx/Vitis-AI)

# Training
* * * 

## Pytorch

[File](./tnq/SRHW_train_pytorch)

### What's done
* Version v1.4
* Trained in house to achievce PSNR: 28.33 and SSIM: 0.9296  [wandb](https://wandb.ai/krislara/SRHW/runs/3190s2ys?workspace=user-krislara)

### File Description
* SRHW.ipynb - Training Code
* pytorchtools.py - Used for Early stopping. Credits: [Bjarte](https://github.com/Bjarten/early-stopping-pytorch)

### Todo
* Reduce the number of model layer and use strategies like knowledge distillation /GAN based methods to imporve accuracy
* Split Images into patches and Super Resolve Patches, use multithreading to make processes parallel at software level. This imporves latency

## Tensorflow

[File](./tnq/SRHW_Quantization/tensorflow)


### What's done
* Version 1.15
* Trained In-house and achieved PSNR:26.476 and SSIM:0.8882. [wandb](https://wandb.ai/krislara/SRHW_Tensorflow/runs/ndifb2q7?workspace=user-krislara)
* Trained as an alternative for Pytorch in Vitis-AI version 1.2 since pytorch compilation did not support DPU for Edge

### File Description
* SRHW_Tensorflow.ipynb - Training code

### Whats Missing in this the repo and how to get it
* local wandb files with best model. Access model (tf_ckpt/model_best.h5) and wandb from above link

### Todo
* Will be deprecated for future releases and hence become stale

## Datasets

[File](./tnq/SRDataset)

* [t91](https://drive.google.com/file/d/1IYLSdFkJb1BNhFcV7W72VkLCFhVulq0i/view?usp=sharing) Training
* [BSD200](https://drive.google.com/file/d/1unp3cmdIpCGvQ1-l7jx8xjAH_GSNCvEi/view?usp=sharing) Training
* [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip) Validation
* [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip) Validation
* [Urban100](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip) Validation
* [Div2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/) Test

## Model Checkpoints

[File](./tnq/SRHW_Quantization/ckpt)

* Note: Checkpoint1_1.pt is Pytorch v1.1 version of checkpoint.pt - used in Vitis AI v1.2

## Quantization
* * *

## Hardware Development
* * * 

## Application Development
* * * 