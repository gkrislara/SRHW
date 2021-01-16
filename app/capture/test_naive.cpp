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

VideoCapture video;

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
    auto start = std::chrono::high_resolution_clock::now();
    for(;;){
        video.read(frame);
        imshow("test",frame);
        temp++;
        if ((temp+ 1)==100){
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<chrono::seconds>(end - start);
            cout << "the display fps is: " << (100 / static_cast<double>(latency.count())) << endl;
        }
        waitKey(1);
    }
    video.release();
    return 0;
}
