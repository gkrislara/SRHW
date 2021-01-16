/*includes*/
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
#include <queue>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace std::chrono;
using namespace cv;

// #define KRENEL_CONV "resnet50_tf_532020_0"

// #define INPUT_NODE "resnet_v1_50_conv1_Conv2D"
// #define OUTPUT_NODE "resnet_v1_50_logits_Conv2D"

class Compare {
 public:
  bool operator()(const pair<int, Mat>& n1, const pair<int, Mat>& n2) const {
    return n1.first > n2.first;
  }
};

VideoCapture video;

bool is_reading =true;
bool is_displaying = true;

queue<pair<int,Mat>> read_queue;
priority_queue<pair<int,Mat>, vector<pair<int,Mat>>, Compare> display_queue;
mutex mtx_queue;
int read_index = 0;
int display_index= 0;
int idx;
float latency=0;
void Read(bool& is_reading){
    auto start = std::chrono::high_resolution_clock::now();
    while(is_reading){
        Mat img;
        
        if(read_queue.size() < 300){
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
            mtx_queue.lock();
            read_queue.push(make_pair(idx,img));
            mtx_queue.unlock();

            mtx_queue.lock();
            display_queue.push(make_pair(idx, img));
            mtx_queue.unlock();

            }
        }
    }


void Display(bool& is_displaying){ 
    auto start = std::chrono::high_resolution_clock::now();
    while(is_displaying){
        if(display_queue.size()>0){
            mtx_queue.lock();
            if(display_index == display_queue.top().first){
                cv::imshow("test",display_queue.top().second);
                display_index++;
                if ((display_index + 1)==100){
                    auto end = std::chrono::high_resolution_clock::now();
                    auto latency = std::chrono::duration_cast<chrono::seconds>(end - start);
                    cout << "the display fps is: " << (100 / static_cast<double>(latency.count())) << endl;
                }
                display_queue.pop();
                mtx_queue.unlock();
                if(waitKey(1) == 'q'){
                    is_reading=false;
                    is_displaying =false;
                    break;
                }
            }
            else
            {
                mtx_queue.unlock();
            }
        }
              
    }
}

int main(int argc,char *argv[])
{
    Mat frame;
    int temp=0;
    
    // int deviceID=0;
    // int apiID=cv::CAP_V4L;
    string filename = argv[1];
    cout<<filename<<endl;
    video.open(filename);
    if (!video.isOpened()){
        cerr<<"Error! Unable to Open Input Device\n";
        return -1;
    }
    // video.set(CV_CAP_PROP_FRAME_WIDTH,1920);
    // video.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    cout<<"Frame Width:"<<video.get(CV_CAP_PROP_FRAME_WIDTH)<<endl;
    cout<<"Frame Height:"<<video.get(CV_CAP_PROP_FRAME_HEIGHT)<<endl;
    
    cout<<"Start Grabbing"<<endl;
    cout<<"Press any key to terminate"<<endl;
    namedWindow("test", CV_WINDOW_AUTOSIZE );
    array<thread,2> threads={
        thread(Read,ref(is_reading)),
        // thread(usleep,150000),
        thread(Display,ref(is_displaying))
    };
    std::cout<<"Threads Created"<<std::endl;
    for(int i=0;i<2;i++){
        threads[i].join();
    }    

    video.release();
    return 0;


}