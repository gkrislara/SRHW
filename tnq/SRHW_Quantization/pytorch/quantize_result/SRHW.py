# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class SRHW(torch.nn.Module):
    def __init__(self):
        super(SRHW, self).__init__()
        self.module_0 = py_nndct.nn.Input() #SRHW::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #SRHW::SRHW/Conv2d[Conv1]/input.2
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #SRHW::SRHW/ReLU[relu]/input.3
        self.module_3 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[1, 5], stride=[1, 1], padding=[0, 2], dilation=[1, 1], groups=32, bias=False) #SRHW::SRHW/Conv2d[DWConv1]/input.4
        self.module_4 = py_nndct.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #SRHW::SRHW/Conv2d[PWConv1]/input.5
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #SRHW::SRHW/ReLU[relu]/input.6
        self.module_6 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[1, 5], stride=[1, 1], padding=[0, 2], dilation=[1, 1], groups=16, bias=False) #SRHW::SRHW/Conv2d[DWConv2]/input.7
        self.module_7 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #SRHW::SRHW/Conv2d[PWConv2]/66
        self.module_8 = py_nndct.nn.Add() #SRHW::SRHW/input.8
        self.module_9 = py_nndct.nn.ReLU(inplace=True) #SRHW::SRHW/ReLU[relu]/input.9
        self.module_10 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=32, bias=False) #SRHW::SRHW/Conv2d[DWConv3]/input.10
        self.module_11 = py_nndct.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #SRHW::SRHW/Conv2d[PWConv3]/input.11
        self.module_12 = py_nndct.nn.ReLU(inplace=True) #SRHW::SRHW/ReLU[relu]/input.12
        self.module_13 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=16, bias=False) #SRHW::SRHW/Conv2d[DWConv4]/input
        self.module_14 = py_nndct.nn.Conv2d(in_channels=16, out_channels=4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #SRHW::SRHW/Conv2d[PWConv4]/114

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_2 = self.module_2(self.output_module_1)
        self.output_module_3 = self.module_3(self.output_module_2)
        self.output_module_4 = self.module_4(self.output_module_3)
        self.output_module_5 = self.module_5(self.output_module_4)
        self.output_module_6 = self.module_6(self.output_module_5)
        self.output_module_7 = self.module_7(self.output_module_6)
        self.output_module_8 = self.module_8(alpha=1, input=self.output_module_2, other=self.output_module_7)
        self.output_module_9 = self.module_9(self.output_module_8)
        self.output_module_10 = self.module_10(self.output_module_9)
        self.output_module_11 = self.module_11(self.output_module_10)
        self.output_module_12 = self.module_12(self.output_module_11)
        self.output_module_13 = self.module_13(self.output_module_12)
        self.output_module_14 = self.module_14(self.output_module_13)
        return self.output_module_14
