#include <fstream>
#include <algorithm>
#include <vector>

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

//get data from file
#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <math.h>

void Test_drawImg(const cv::Mat c); 
void  Getluminance(const cv::Mat mat,float lumdis[]);
float GetWHFeature(const cv::Mat mat);
void GetHogFeature(const cv::Mat &src,std::vector<float> &vHogFeature);
