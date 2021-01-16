#!/bin/bash

CXX=${CXX:-g++}
$CXX -std=c++11 -O3 -I /home/htic-broncho/TRD20/SRHW_SDK/vai_dpu/sysroots/aarch64-xilinx-linux/usr/include/drm srhw_dpu.cpp -o srhw -ln2cube -lhineon -lopencv_videoio -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_core -pthread -lxrt_core

