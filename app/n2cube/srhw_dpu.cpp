#include "./drmhdmi.hpp"
#include <dnndk/dnndk.h>
#include <dnndk/n2cube.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <queue>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <cstring>

/*vart*/
// #include <glog/logging.h>

// #include <algorithm>
// #include <cmath>
// #include <functional>
// #include <iomanip>
// #include <iostream>
// #include <numeric>
// #include <vitis/ai/env_config.hpp>
// #include <xir/graph/graph.hpp>

// #include "vart/dpu/dpu_runner_ext.hpp"
// #include "vart/dpu/vitis_dpu_runner_factory.hpp"
// #include "vart/mm/host_flat_tensor_buffer.hpp"
// #include <memory>
// #include "vart/tensor_buffer.hpp"

#define KERNEL_NAME "SRHWdb"
#define INPUT_NODE "conv2d_Conv2D"
#define OUTPUT_NODE "conv2d_4_Conv2D"

using namespace std;
using namespace std::chrono;

#define SHOWTIME
#ifdef SHOWTIME
#define _T(func)                                                          \
  {                                                                       \
    auto _start = system_clock::now();                                    \
    func;                                                                 \
    auto _end = system_clock::now();                                      \
    auto duration = (duration_cast<milliseconds>(_end - _start)).count(); \
    string tmp = #func;                                                   \
    tmp = tmp.substr(0, tmp.find('('));                                   \
    cout << "[TimeTest]" << left << setw(30) << tmp;                      \
    cout << left << setw(10) << duration << "ms" << endl;                 \
  }
#else
#define _T(func) func;
#endif



// dnndk api

const int out_size=1920*1080*4;

static float outdata[out_size]={};


void PixelShuffle(cv::Mat HRes,float *data,int upscale, vector<int> inshape,float scale)
{
    //include address of HR as argument
    std::cout<<"HWC:"<<inshape[0]<<"x"<<inshape[1]<<"x"<<inshape[2]<<std::endl;
    int r=upscale;
    int r2=r*r;
    int out_height=inshape[0]*r;
    int out_width=inshape[1]*r;
    int C=inshape[2]/r2;
    std::cout<<"HWC:"<<out_height<<"x"<<out_width<<"x"<<C<<std::endl;
    std::cout<<"HWC:"<<floor((out_height-1)/r)<<"x"<<floor((out_width-1)/r)<<"x"<<C*r*((out_width-1)%r)+C*((out_height-1)%r)<<std::endl;
    int ix=0;
    int iy=0;
    int ic=0;    
    int index=0;
    int max=0;
    cv::Mat HR=cv::Mat::zeros(out_height,out_width,CV_32F);
    for (int x=0;x<out_height;x++){
        for(int y=0;y<out_width;y++){
                ix=floor(x/r);
                iy=floor(y/r);
                ic=C*r*(y%r)+C*(x%r);
                index=ix*inshape[1]*inshape[2]+iy*inshape[2]+ic;
                if(index<out_size){
                    HRes.at<float>(x,y)=data[index]; 
                    //std::cout<<HR.at<float>(x,y)<<" ";// TODO channels is 1 refactor for dynamic number of channels
                }
                else
                {
                    std::cout<<"Array out of Bound"<<std::endl;
                }
                
        }
    }
}


int main(int argc,char **argv)
{


dpuOpen();

DPUKernel *kernelConv;
DPUTask *taskConv;

kernelConv=dpuLoadKernel(KERNEL_NAME);
taskConv=dpuCreateTask(kernelConv,1);

int height=dpuGetInputTensorHeight(taskConv,INPUT_NODE,0);
int width=dpuGetInputTensorWidth(taskConv,INPUT_NODE,0);
int channel=dpuGetInputTensorChannel(taskConv,INPUT_NODE,0);

cv::Mat img = cv::Mat::ones(height, width, CV_32F);
std::cout<<"INPUT Size:"<<width<<"x"<<height<<"x"<<channel<<"\n";



int8_t *inputAddr=dpuGetInputTensorAddress(taskConv,INPUT_NODE,0);
float inScale = dpuGetInputTensorScale(taskConv,INPUT_NODE,0);
float outScale= dpuGetOutputTensorScale(taskConv,OUTPUT_NODE,0);

std::cout<<"ScaleFix In:"<<inScale<<std::endl;
std::cout<<"ScaleFix Out:"<<outScale<<std::endl;
// cv::normalize(img,img,0,1,cv::NORM_MINMAX,CV_32F);

for (int idx_h=0;idx_h<img.rows;idx_h++){
    for(int idx_w=0;idx_w<img.cols;idx_w++){
            // if(idx_w<10){
            //     std::cout<<img.at<float>(idx_h,idx_w)<<"\n";
            // }
            inputAddr[idx_h*img.cols+idx_w]=img.at<float>(idx_h,idx_w)*inScale;
        }
    }

uint32_t core= dpuGetTaskAffinity(taskConv);
std::cout<<"Before Setting CORE:"<<core<<std::endl;
int success=dpuSetTaskAffinity(taskConv,1);
core= dpuGetTaskAffinity(taskConv);
std::cout<<"After Setting CORE:"<<core<<std::endl;

if (success == 0){
    dpuRunTask(taskConv);
}
else{
    std::cout<<"CORE NOT SET PROPERLY. NOT RUNNING DPU"<<std::endl;
}


int prof = dpuGetTaskProfile(taskConv);
std::cout<<"Latency:"<<prof<<" us"<<std::endl;
int out_height=dpuGetOutputTensorHeight(taskConv,OUTPUT_NODE,0);
int out_width=dpuGetOutputTensorWidth(taskConv,OUTPUT_NODE,0);
int out_channel=dpuGetOutputTensorChannel(taskConv,OUTPUT_NODE,0);
std::cout<<"OUTPUT Size:"<<out_width<<"x"<<out_height<<"x"<<out_channel<<std::endl;
int total_size=out_height*out_width*out_channel;
cv::Mat HR=cv::Mat::zeros(out_height,out_width,CV_32F);
cv::Mat SR=cv::Mat::zeros(out_height,out_width,CV_32F);
vector<int> out_shape{out_height,out_width,out_channel};

std::cout<<"Printing first 10 values before fetch...."<<"\n";
for(int i=0;i<10;i++){
    std::cout<<outdata[i]<<" ";
}
std::cout<<std::endl;

dpuGetOutputTensorInHWCFP32(taskConv,OUTPUT_NODE,outdata,total_size);

std::cout<<"Data Received!!"<<"\n";
std::cout<<"Printing first 10 values...."<<"\n";
for(int i=0;i<10;i++){
    std::cout<<outdata[i]<<" ";
}
std::cout<<std::endl;

int Ups=sqrt(out_channel/channel);

std::cout<<"UPSCALE:"<<Ups<<std::endl;
PixelShuffle(HR,outdata,Ups,out_shape,outScale);
HR.convertTo(SR,CV_8U);

std::cout<<"SR SIZE:"<<HR.rows<<"x"<<HR.cols<<"x"<<HR.channels();

std::cout<<"Printing first 10 values of SR..."<<"\n";
for(int i=0;i<10;i++){
    std::cout<<HR.at<float>(0,i)<<" ";
}
std::cout<<std::endl;
cv::imwrite("HR.png",HR);
dpuDestroyTask(taskConv);
dpuDestroyKernel(kernelConv);
dpuClose();
return 0;

}


/*vart api */

// static std::unique_ptr<vart::TensorBuffer> create_cpu_flat_tensor_buffer(
//     const xir::Tensor* tensor) {
//   return std::make_unique<vart::mm::HostFlatTensorBuffer>(tensor);
// }


// int main(int argc,char** argv)
// {

//     const auto filename="dpu_SRHW.elf";
//     const auto kernel_name=std::string("SRHW");

//     auto runner=vart::dpu::DpuRunnerFactory::create_dpu_runner(filename,kernel_name);

//     auto input_scale =dynamic_cast<vart::dpu::DpuRunnerExt*>(runner.get())->get_input_scale();
//     auto output_scale = dynamic_cast<vart::dpu::DpuRunnerExt*>(runner.get())->get_output_scale();

//     auto input_tensors = runner->get_input_tensors();
//     auto input_tensor = input_tensors[0];
//     auto height = input_tensor->get_dim_size(1);
//     auto width = input_tensor->get_dim_size(2);
//     auto channel = input_tensor->get_dim_size(3);

//     std::cout<<"INPUT Size:"<<height<<"x"<<width<<"x"<<channel<<std::endl;

//     cv::Mat img = cv::Mat::ones(height, width, CV_32F);

//     auto input_tensor_buffer = create_cpu_flat_tensor_buffer(input_tensor);    

//     auto output_tensors = runner->get_output_tensors();

//     auto output_tensor = output_tensors[0];
//     auto out_height=output_tensor->get_dim_size(1);
//     auto out_width=output_tensor->get_dim_size(2);
//     auto out_channels=output_tensor->get_dim_size(3);

//     std::cout<<"OUTPUT Size:"<<out_height<<"x"<<out_width<<"x"<<out_channels<<std::endl;

//     auto output_tensor_buffer = create_cpu_flat_tensor_buffer(output_tensor);

//     uint64_t data_in = 0u;
//     size_t size_in = 0u;
//     std::tie(data_in,size_in)=input_tensor_buffer->data(std::vector<int>{0,0,0,0});

//     signed char* data=(signed char*)data_in;

//     for (int idx_h=0;idx_h<img.rows;idx_h++){
//     for(int idx_w=0;idx_w<img.cols;idx_w++){
//             // if(idx_w<10){
//             //     std::cout<<img.at<float>(idx_h,idx_w)<<"\n";
//             // }
//             data[idx_h*img.cols+idx_w]=img.at<float>(idx_h,idx_w)*input_scale[0];
//         }
//     }

//     auto v = runner->execute_async({input_tensor_buffer.get()},{output_tensor_buffer.get()});

//     auto status= runner->wait((int)v.first, -1);

//     if(status == 0){
//         std::cout<<"RUN SUCESSFUL!"<<std::endl;
//     }
//     else
//     {
//         std::cout<<"DPU RUN ERROR! CHECK YOUR CODE.."<<std::endl;
//     }
    
//     uint64_t out_data=0u;
//     size_t out_size = 0u;
//     std::tie(out_data,out_size)=output_tensor_buffer->data(std::vector<int>{0,0,0,0});
//     signed char* data_c= (signed char*)out_data;
//     auto out =std::vector<float>(out_size);

//     std::transform(data_c,data_c + out_size, out.begin(),[output_scale](signed char v){
//         return ((float)v)*output_scale[0];
//     });

//     std::cout<<"Printing first 10 values ...."<<"\n";
//     for(int i=0;i<10;i++){
//         std::cout<<out[i]<<" ";
//     }
//     std::cout<<std::endl;

// }

