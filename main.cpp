#include <iostream>
#include "Classify.h"

int main()
{
    const char* filedir = "/data/TB/TBdata/training/ori/png/041189_1.png";
    cv::Mat matsrc = cv::imread(filedir, IMREAD_GRAYSCALE);
    int bFlag = GetClassfy(matsrc);
    std::cout << bFlag << std::endl;
    return 0;
}

   
