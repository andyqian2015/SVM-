
#include <iostream>
#include <fstream>

#include <vector>
#include <algorithm>
#include <cstring>
#include <numeric>

//create file
#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sstream>
#include <math.h>

//time
#include <sys/time.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv; 
using namespace cv::ml;

void Getluminance(const cv::Mat mat,float lumdis[]);
void GetHogFeature(const cv::Mat &src,std::vector<float> &vHogFeature);
