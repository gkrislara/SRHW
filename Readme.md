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
* Ethernet Cables for connectivity between systems (can be used if theres no monitor to display on host system)

## Prerequisites
* Knowledge on FPGA and Integrating IPs in Vivado
* Knowledge on Computer Architecture and Embedded Systems
* Linux Operating system - Petalinux/Yocto
* Experience in building Deep learning models
* Programming Languages: Cpp - 17 and Python - 3.8+
* Frameworks/libraries: Opencv,Numpy,Tensorflow and Pytorch

This documentation describes the state the repo and is divided into four sections

1. [Training](#markdown-header-Training)
2. [Quantization](#markdown-header-Quantization)
3. [Hardware Development](#markdown-header-Hardware Development)
4. [Application Development](#markdown-header-Application Development)

For complete information about the experiments and iterations follow [here](https://htic.atlassian.net/wiki/spaces/~11214967/pages/93487268/Daily+Work+Update) from 29/10/2020 to 15/01/2021

For How-To's and project flows and other features follow [Vitis AI User Documentation](https://www.xilinx.com/html_docs/vitis_ai/1_3/index.html) and [Vitis-AI Github](https://github.com/Xilinx/Vitis-AI)

# Training
* * * 