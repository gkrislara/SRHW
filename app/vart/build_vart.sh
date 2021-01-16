#/bin/sh
CXX=${CXX:-g++}
$CXX -std=c++17 -O3 -I /home/htic-broncho/TRD20/SRHW_SDK/vai_dpu/sysroots/aarch64-xilinx-linux/usr/include/drm -o srhw_vart srhw_vart.cpp -lglog -lvart-mem-manager -lxir -lunilog -lvart-buffer-object -lvart-runner -lvart-util -lvart-xrt-device-handle -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lvart-dpu-runner -lvart-dpu-controller -lopencv_highgui -lopencv_videoio -lpthread
