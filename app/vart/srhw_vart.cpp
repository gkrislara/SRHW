/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <xir/graph/graph.hpp>

#include "vart/runner_ext.hpp"
#include "vart/dpu/vitis_dpu_runner_factory.hpp"
#include "vart/mm/host_flat_tensor_buffer.hpp"
#include "vart/tensor_buffer.hpp"

static cv::Mat read_image(const std::string& image_file_name) {
  // read image from a file
  auto input_image = cv::imread(image_file_name,-1 );
//   CHECK(!input_image.empty()) << "cannot load " << image_file_name;
  return input_image;
}

std::vector<cv::Mat> setImageY(const cv::Mat& image, void* data1,float scale,int ups
                                //std::vector<double> &bound
                                ){

  signed char* data = (signed char*)data1; //upcast
  cv::cvtColor(image,image,cv::COLOR_BGR2YUV);
  cv::Mat channels[3];
//conversion
  cv::split(image,channels);
  // cv::minMaxLoc(channels[0],&bound[0],&bound[1]);

  for (auto row = 0; row < channels[0].rows; row++) {
    for (auto col = 0; col < channels[1].cols; col++) {
      auto v = channels[0].at<u_char>(row, col);
      auto Y = (float)v / 255.0;
      auto scaleFixY = Y * scale ;
      data[row*image.cols+col] = (int)scaleFixY;//(int)scaleFixY;
    }
  }

  cv::resize(channels[1],channels[1],cv::Size(channels[1].cols * ups,channels[1].rows * ups),cv::INTER_CUBIC);
  cv::resize(channels[2],channels[2],cv::Size(channels[2].cols * ups,channels[2].rows * ups),cv::INTER_CUBIC);

  std::vector<cv::Mat> bicub_ch={channels[1],channels[2]};
  return bicub_ch;
}

cv::Mat PixelShuffle(const std::vector<unsigned char>& conv, vector<int> inshape,int upscale)
{
    //include address of HR as argument
    // std::cout<<"HWC:"<<inshape[0]<<"x"<<inshape[1]<<"x"<<inshape[2]<<std::endl;
    int r=upscale;
    int r2=r*r;
    int out_height=inshape[0]*r;
    int out_width=inshape[1]*r;
    int C=inshape[2]/r2;

    int ix=0;
    int iy=0;
    int ic=0;    
    int index=0;
    int max=0;
    int out_size = inshape[0]*inshape[1]*inshape[2];
    cv::Mat HR=cv::Mat::zeros(out_height,out_width,CV_8U);
    for (int x=0;x<out_height;x++){
        for(int y=0;y<out_width;y++){
                ix=floor(x/r);
                iy=floor(y/r);
                ic=C*r*(y%r)+C*(x%r);
                index=ix*inshape[1]*inshape[2]+iy*inshape[2]+ic;
                if(index<out_size){
                    HR.at<unsigned char>(x,y)=conv[index]; 
                }
                else
                {
                    std::cout<<"Array out of Bound"<<std::endl;
                    exit(0);
                }
                
        }
    }
    return HR;
}


static std::unique_ptr<vart::TensorBuffer> create_cpu_flat_tensor_buffer(
    const xir::Tensor* tensor) {
  return std::make_unique<vart::mm::HostFlatTensorBuffer>(tensor);
}


static std::vector<unsigned char> convert_fixpoint_to_float(
    vart::TensorBuffer* tensor_buffer, float scale
    // std::vector<double> &boundary
    ) {
  uint64_t data = 0u;
  size_t size = 0u;
  std::tie(data, size) = tensor_buffer->data(std::vector<int>{0,0, 0, 0});
  signed char* data_c = (signed char*)data;

  auto ret = std::vector<unsigned char>(size);

  // int ipmin=boundary[0];
  // int ipmax=boundary[1];
  transform(data_c, data_c + size, ret.begin(),
            [scale](signed char v) {
              // [min,max,ipmin,ipmax]return ipmin+((float(v)-min)/(max-min))*(ipmax-ipmin);
              return (int)std::min(std::max(v * scale *255,0.0f),255.0f);
             });
  return ret;
}


cv::Mat post_process(
    vart::TensorBuffer* tensor_buffer,vector<cv::Mat> &SRcromas,float scale, int ups
    //,std::vector<double> &bound
    ){
  auto conv_output = convert_fixpoint_to_float(tensor_buffer, scale);
  //take output of conv shape
  int conv_channels = conv_output.size()/(SRcromas[0].rows/ups * SRcromas[0].cols/ups);
  vector<int> conv_shape = {SRcromas[0].rows/ups,SRcromas[0].cols/ups,conv_channels};
  cv::Mat SR_Y = PixelShuffle(conv_output,conv_shape,ups);
  std::vector<cv::Mat> channels={SR_Y,SRcromas[0],SRcromas[1]};
  cv::Mat SRes= cv::Mat::zeros(SR_Y.rows,SR_Y.cols,CV_8UC3);
  cv::merge(channels,SRes);
  cv::cvtColor(SRes,SRes,cv::COLOR_YUV2BGR);
  return SRes;
}


class Compare {
 public:
  bool operator()(const pair<int, cv::Mat>& n1, const pair<int, cv::Mat>& n2) const {
    return n1.first > n2.first;
  }
};

cv::VideoCapture video;

bool is_reading = true;
bool is_running =true;
bool is_running2 =true;
bool is_displaying =true;

std::queue<pair<int,cv::Mat>> read_queue;
std::priority_queue<std::pair<int,cv::Mat>,std::vector<std::pair<int,cv::Mat>>,Compare> display_queue;
mutex mtx_read_queue,mtx_display_queue;
int read_index = 0;
int display_index= 0;

void Read(bool& is_reading){
    int idx;
    auto start = std::chrono::high_resolution_clock::now();
    while(is_reading){
        cv::Mat img;
        
        if(read_queue.size() < 1000){
            bool success=video.read(img);
            if (!success){
                std::cout<<"Finish reading the Video."<<std::endl;
                is_reading=false;
                break;
            }
            idx=read_index++;
            if ((idx + 1)==100){
                auto end = std::chrono::high_resolution_clock::now();
                auto latency = std::chrono::duration_cast<chrono::seconds>(end - start);
                cout << "the fps is: " << (100 / static_cast<double>(latency.count())) << endl;
            }
            mtx_read_queue.lock();
            read_queue.push(make_pair(idx,img));
            mtx_read_queue.unlock();
            // mtx_display_queue.lock();
            // display_queue.push(make_pair(idx,img));
            // mtx_display_queue.unlock();
            }
        }
    }


void Display(bool& is_displaying){ 
    auto start = std::chrono::high_resolution_clock::now();
    while(is_displaying){
        mtx_display_queue.lock();
        if(display_queue.size()<=0){
          if(is_running){
            mtx_display_queue.unlock();
          }
          else
          {
            is_running=false;
            break;
          }
          
        }
        else if(display_index == display_queue.top().first){
                  cv::imshow("SR",display_queue.top().second);

                display_index++;
                if ((display_index + 1)==100){
                    auto end = std::chrono::high_resolution_clock::now();
                    auto latency = std::chrono::duration_cast<chrono::seconds>(end - start);
                    std::cout << "the display fps is: " << (100 / static_cast<double>(latency.count())) << std::endl;
                }
                display_queue.pop();
                mtx_display_queue.unlock();
                if(cv::waitKey(1) == 'q'){
                    is_reading=false;
                    is_running=false;
                    is_displaying =false;
                    break;
                }
            }

        else
            {
                mtx_display_queue.unlock();
            }
         }
        }          

// add feature to include json file param filename and kernel

void Run(bool &is_running,
          vart::Runner* run, vart::TensorBuffer* input_tensor, vart::TensorBuffer* output_tensor, 
          void* data1,int iscale, float oscale, int upscale_factor){
        while(is_running){
          int index;
          cv::Mat img;
          mtx_read_queue.lock();
          if(read_queue.empty()){
            mtx_read_queue.unlock();
            if(is_reading){
              continue;
            }
            else
              is_running =false;
              break;
            }
          else
          {
            index = read_queue.front().first;
            img=read_queue.front().second;
            read_queue.pop();
            mtx_read_queue.unlock();
          }

          std::vector<cv::Mat> SRCroma = setImageY(img, (void*)data1, iscale, upscale_factor);
          auto _start = std::chrono::high_resolution_clock::now();
          ios_base::sync_with_stdio(false); 
          auto v = run->execute_async({input_tensor},
                                   {output_tensor});
          auto _end = std::chrono::high_resolution_clock::now();
          auto duration = (chrono::duration_cast<chrono::microseconds>(_end - _start)).count();                               
          auto status = run->wait((int)v.first, -1);
          if(status != 0){
              std::cout<< "failed to run dpu with status:"<<status<<std::endl;
              exit(0);
          } 
          // cv::Mat SR = post_process(output_tensor, SRCroma, oscale,upscale_factor);
          mtx_display_queue.lock();
          display_queue.push(make_pair(index, post_process(output_tensor, SRCroma, oscale,upscale_factor)));
          mtx_display_queue.unlock();    
        }

      }


int main(int argc, char* argv[]) {
  
    // const auto image_file_name = std::string(argv[1]);  // std::string(argv[2]);
    const auto video_file_name = std::string(argv[1]);
    const auto filename = std::string(argv[2]);//"SRHW.xmodel";            //
    const auto kernel_name = "subgraph_SRHW__Conv1_weight"; //std::string(argv[3]);
    std::cout<<"File:"<<filename<<std::endl<<"Kernel:"<<kernel_name<<std::endl;
    std::cout<<"Video_file:"<<video_file_name<<std::endl;

    auto runner =
      vart::dpu::DpuRunnerFactory::create_dpu_runner(filename, kernel_name);
    // auto runner2 =
    //   vart::dpu::DpuRunnerFactory::create_dpu_runner(filename, kernel_name);
    

    // cv::Mat RGB_In_Image = read_image(image_file_name);

    video.open(video_file_name);
    if (!video.isOpened()){
      std::cerr<<"ERROR! Unable to Open Input Device\n";
      return -1;
    }
    video.set(CV_CAP_PROP_FRAME_WIDTH,1920);
    video.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    
    std::cout<<"Frame Width:"<<video.get(CV_CAP_PROP_FRAME_WIDTH)<<std::endl;
    std::cout<<"Frame Height:"<<video.get(CV_CAP_PROP_FRAME_HEIGHT)<<std::endl;
    cv::Mat RGB_In_Image;

        auto input_tensors = runner->get_input_tensors();
        // auto input_tensors2 = runner2->get_input_tensors();
    auto input_scale = vart::get_input_scale(input_tensors);
    if(input_tensors.size() != 1u){
        std::cout<<"Input Tensor in not 1 please verify the model"<<std::endl;
        return 0;
    }

    auto input_tensor = input_tensors[0];
    // auto input_tensor2 = input_tensors2[0];
    auto height = input_tensor->get_shape().at(1);
    auto width = input_tensor->get_shape().at(2);
    auto in_channels = input_tensor -> get_shape().at(3);


    auto input_tensor_buffer = create_cpu_flat_tensor_buffer(input_tensor);
    // auto input_tensor_buffer2 = create_cpu_flat_tensor_buffer(input_tensor2);
    // prepare output tensor buffer
    auto output_tensors = runner->get_output_tensors();
    // auto output_tensors2 = runner2->get_output_tensors();
    auto output_scale = vart::get_output_scale(output_tensors);
    if(output_tensors.size() != 1u){
        std::cout<<"Output Tensor in not 1 please verify the model"<<std::endl;
        return 0;
    }
    auto output_tensor = output_tensors[0];
    // auto output_tensor2 = output_tensors2[0];
    auto out_channels = output_tensor-> get_shape().at(3);

    auto output_tensor_buffer = create_cpu_flat_tensor_buffer(output_tensor);
    // auto output_tensor_buffer2 = create_cpu_flat_tensor_buffer(output_tensor2);


    int upscale= sqrt(out_channels/in_channels);
    uint64_t data_in = 0u;
    size_t size_in = 0u;
    std::tie(data_in, size_in) =
        input_tensor_buffer->data(std::vector<int>{0,0,0,0});
    
    // uint64_t data_in2 = 0u;
    // size_t size_in2 = 0u;
    // std::tie(data_in2, size_in2) =
    //     input_tensor_buffer2->data(std::vector<int>{0,0,0,0});

    // std::vector<double> boundary={0};



    std::cout<<"Start Grabbing"<<std::endl;
    std::cout<<"Press any key to terminate"<<std::endl;

    cv::namedWindow("SR",CV_WINDOW_AUTOSIZE);
    array<thread,4> threads = {
      thread(Read,std::ref(is_reading)),
      thread(Run,std::ref(is_running),runner.get(),input_tensor_buffer.get(),output_tensor_buffer.get(),
            (void*)data_in,input_scale[0],output_scale[0],ref(upscale)),
      thread(Run,std::ref(is_running),runner.get(),input_tensor_buffer.get(),output_tensor_buffer.get(),
            (void*)data_in,input_scale[0],output_scale[0],ref(upscale)),
      // thread(Run,std::ref(is_running),runner.get(),input_tensor_buffer.get(),output_tensor_buffer.get(),
      //       (void*)data_in,input_scale[0],output_scale[0],ref(upscale)),
      thread(Display,ref(is_displaying))
    };
    std::cout<<"Threads Created"<<std::endl;

    for(int i=0;i<4;i++){
      threads[i].join();
    }


    // std::string save=argv[1];
  return 0;
}

//DEBUG 

    // for(;;)
    // {
    //   bool success = video.read(RGB_In_Image);
    //   if(!success){
    //     std::cerr<<"ERROR! Blank Frame!\n";
    //     exit(0);
    //   }
    //   //logic
    //   std::vector<cv::Mat> SRCroma = setImageY(RGB_In_Image, (void*)data_in, input_scale[0], upscale);

    //   auto _start = std::chrono::high_resolution_clock::now();
    //   ios_base::sync_with_stdio(false); 
    //   auto v = runner->execute_async({input_tensor_buffer.get()},
    //                                {output_tensor_buffer.get()});
    //   auto _end = std::chrono::high_resolution_clock::now();
    //   auto duration = (chrono::duration_cast<chrono::microseconds>(_end - _start)).count();                               
    //   auto status = runner->wait((int)v.first, -1);
    //   if(status != 0){
    //       std::cout<< "failed to run dpu with status:"<<status<<std::endl;
    //       exit(0);
    //   } 

    //   static cv::Mat SR = post_process(output_tensor_buffer.get(), SRCroma, output_scale[0],upscale);

    //   cv::imshow("SR",SR);
    //   char c= (char)cv::waitKey(1);
    //   if (c==27) break;

    // }


// std::string type2str(int type) {
//   string r;

//   uchar depth = type & CV_MAT_DEPTH_MASK;
//   uchar chans = 1 + (type >> CV_CN_SHIFT);

//   switch ( depth ) {
//     case CV_8U:  r = "8U"; break;
//     case CV_8S:  r = "8S"; break;
//     case CV_16U: r = "16U"; break;
//     case CV_16S: r = "16S"; break;
//     case CV_32S: r = "32S"; break;
//     case CV_32F: r = "32F"; break;
//     case CV_64F: r = "64F"; break;
//     default:     r = "User"; break;
//   }

//   r += "C";
//   r += (chans+'0');

//   return r;
// }


// -----const auto kernel_name = "subgraph_SRHW__Conv1_weight"; //std::string(argv[3]);
// std::cout<<"File:"<<filename<<std::endl<<"Kernel:"<<kernel_name<<std::endl;
// auto runner = ---------

// std::cout<<"runner created \n"<<runner.get()<<"\n";

// std::cout<<"Image Read:"<<RGB_In_Image.at<cv::Vec3b>(0,0)[0]<<"\n";
// std::cout<<"Img type:"<<type2str(RGB_In_Image.type())<<std::endl;
// prepare input tensor buffer

// std::cout<<"input tensor initialised\n";


// std::cout<<"In_Dimensions:"<<height<<"x"<<width<<"x"<<in_channels<<std::endl;

// std::cout<<"Out_Channels:"<<out_channels<<std::endl;

// std::cout<<"Buffer created; before setting image"<<std::endl;
// std::cout<<"Data Pointer:"<<(void*)data_in<<std::endl;

// std::cout<<"Y min:"<<boundary[0]<<std::endl;
// std::cout<<"Y max:"<<boundary[1]<<std::endl;

// std::cout<<(int)SRCroma[0].at<u_char>(0,0)<<std::endl;
// std::cout<<(SRCroma[0].rows * SRCroma[0].cols)<<std::endl;
// start the dpu


// std::cout<<"Duration:"<<duration<<std::endl;


// std::string name = save.substr(save.find_last_of(".") + 1);

// cv::imwrite("SR_"+save,SR);
// std::cout<<"Image Written"<<std::endl;


// ---   cv::minMaxLoc(channels[0],&bound[0],&bound[1]);
// std::cout<<"minval:"<<bound[0]<<std::endl;
// std::cout<<"maxval:"<<bound[1]<<std::endl;
// for (auto row = 0; row < channels[0].rows; row++) { -------

//----  data[row*image.cols+col] = (int)scaleFixY;//(int)scaleFixY;}}
// std::cout<<"Image set and U,V extracted"<<std::endl;
// std::cout<<"Data:"<<int(data[0])<<std::endl;
// cv::resize(channels[1],channels[1],cv::Size(channels[1].cols * ups,channels[1].rows * ups),cv::INTER_CUBIC); -------

// -----cv::resize(channels[2],channels[2],cv::Size(channels[2].cols * ups,channels[2].rows * ups),cv::INTER_CUBIC);
// std::cout<<"Bicubic Interpolation Done"<<std::endl;
// std::vector<cv::Mat> bicub_ch={channels[1],channels[2]}; --------


// --- int C=inshape[2]/r2;
// std::cout<<"HWC:"<<out_height<<"x"<<out_width<<"x"<<C<<std::endl;
// std::cout<<"HWC:"<<floor((out_height-1)/r)<<"x"<<floor((out_width-1)/r)<<"x"<<C*r*((out_width-1)%r)+C*((out_height-1)%r)<<std::endl;
// int ix=0; -----


//-----auto ret = std::vector<unsigned char>(size);
//min-max-std
// int max=-1000000;
// int min=1000000;
// int mean=0;
// int dev_idx=0;
// if(size % 2 == 0){
//   dev_idx = size/2;
// }
// else
// {
//   dev_idx=(size+1)/2;
// }
// int c=0;

// int median=int(*(data_c+dev_idx));
// for(int i=0;i<size;i++)
// {
//   int d=int(*(data_c+i));
//   mean+=d;
//   if(max<d){
//     max=d;
//   }
//   if(min>d){
//     min=d;
//   }
//   if(d<0){
//     c++;
//   }
// }
// mean/=size;
// float var=0;
// for(int i=0;i<size;i++){
//   int d=int(*(data_c+i));
//   var += pow(d-mean,2);
// }
// var/=size;
// float stdev =sqrt(var);
// std::cout<<"Max:"<<max<<std::endl;
// std::cout<<"Min:"<<min<<std::endl;
// std::cout<<"Scale:"<<scale<<std::endl;  
// std::cout<<"Avg:"<<mean<<std::endl;
// std::cout<<"median:"<<median<<std::endl;
// std::cout<<"std:"<<stdev<<std::endl;  
// std::cout<<"NumNegative:"<<c<<std::endl;
// std::cout<<"bound:"<<boundary[0]<<" "<<boundary[1]<<std::endl;
//----- int ipmin=boundary[0];


// -----return (int)std::min(std::max(v * scale *255,0.0f),255.0f); });
// std::cout<<"ret min: "<<(int)*min_element(ret.begin(),ret.end())<<std::endl;
// std::cout<<"ret max: "<<(int)*max_element(ret.begin(),ret.end())<<std::endl;
// ----   return ret;