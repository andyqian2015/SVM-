#include "Head.h"
#include "Classify.h"


//获取HOG特征
void GetHogFeature(const cv::Mat &src,std::vector<float> &vHogFeature)
{
    HOGDescriptor hog = HOGDescriptor(Size(64,64),Size(16,16),Size(8,8),Size(8,8),3);
    int DescriptorDim = 0;
    std::vector<float> descriptors;//HOG描述子向量
    hog.compute(src,descriptors);
    DescriptorDim = descriptors.size();
    vHogFeature.resize(DescriptorDim);
    for(int i=0; i<DescriptorDim; i++)
        vHogFeature[i] = descriptors[i];
}

void SetFeature(const cv::Mat mat,std::vector<float> &vFeature)
{
    float lumdis[mat.cols] = {0};
    Getluminance(mat,lumdis);
    for(int i= 0 ; i < mat.rows; ++i){
        vFeature.push_back(lumdis[i]);
    }
    //将宽高比放到特征中去
    //将HOG特征加入进去
    cv::Mat dst;
    cv::resize(mat,dst,cv::Size(64,64));
    std::vector<float>vHogFeature(0);
    GetHogFeature(dst,vHogFeature);
    for(unsigned int i = 0 ; i < vHogFeature.size();++i)
        vFeature.push_back(vHogFeature[i]);

}


//获取在y轴方向上的亮度点
void Getluminance(const cv::Mat mat,float lumdis[])
{
    int nWidth  = mat.cols;
    cv::Mat colvec;
    cv::reduce(mat,colvec,0,REDUCE_SUM,CV_32F);
    cv::Mat m1(1,nWidth,CV_64FC1,Scalar(0));
    for(int i = 0 ; i < nWidth;++i)
    {
        m1.at<double>(0,i) = (double)colvec.at<uchar>(0,i);
    }    
    cv::Mat tmp_m, tmp_sd;
    cv::meanStdDev(colvec,tmp_m,tmp_sd);
    cv::normalize(m1,m1); 
    for(int i = 0;i < nWidth;++i)
    {
        lumdis[i] = m1.at<double>(0,i);
    }
}


int SVM_Classfy(const std::vector<float>& vFeature)
{
    int nFeatureSize = vFeature.size();
    Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(
            "/home/dongdong/qian/ribtagclassfy/car.xml");
    float test[nFeatureSize] = {0};
    for(int i = 0 ; i < nFeatureSize;++i)
    {
        test[i] = vFeature[i];
    }
    cv::Mat sample = cv::Mat(1, nFeatureSize, CV_32FC1,test);
    float result = svm->predict(sample);   //预测分区域
    if (result == 1.0)
        return 1;
    else if(result == -1.0)
        return -1;

    return 0;
}

cv::Mat Normalization(const cv::Mat &srcmat)
{
   cv::Mat dstMat;
   cv::Mat mat = cv::Mat(512, 512,CV_8UC1);
   cv::resize(srcmat, dstMat, cv::Size(512, 512));
   int nRows = mat.rows;
   int nCols = mat.cols;
   double maxV =0,minV = 0;
   Point maxP,minP;
   minMaxLoc(dstMat,&minV,&maxV,&minP,&maxP);
   for (int i = 0 ; i < nRows; ++i)
   {   
     for(int j = 0; j < nCols;++j)
       mat.at<uchar>(i,j) = (uchar)((unsigned short)((dstMat.at<unsigned short>(i,j)-minV)/(maxV - minV)));
   }
   return mat;
}


int GetClassfy(const cv::Mat &m)
{
    cv::Mat mat = Normalization(m);
    std::vector<float> vFeature(0);    
    SetFeature(mat, vFeature);
    std::cout << vFeature.size() << std::endl;
    int id = SVM_Classfy(vFeature);
    return id;  
}


/*
int main()
{
    const char* filedir = "/data/TB/TBdata/training/ori/png/041189_1.png";
    cv::Mat matsrc = cv::imread(filedir, IMREAD_GRAYSCALE);
    int bFlag = GetClassfy(matsrc);
    std::cout << bFlag << std::endl;
    return 0;
}
*/
