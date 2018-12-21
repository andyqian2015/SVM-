#include "Head.h"

int SVM_Classfy(const std::vector< std::vector<float> > &vvFeature, int lables[]) {
     
     
     int nTrainSize = vvFeature.size();
     int nFeatureSize = vvFeature[0].size();
      
     if(nFeatureSize <1 || nTrainSize < 1)
     {
       std::cout << "Train Sample is error" << std::endl;
       return 0;
     }
      
     std::cout << nTrainSize << " " << nFeatureSize << std::endl;
          
     cv::Mat lablesMat(nTrainSize, 1, CV_32SC1, lables);
     float trainingData[nTrainSize][nFeatureSize] = {0};
     for(int i = 0 ; i < nTrainSize;++i)
     {
       for(int j = 0 ; j < nFeatureSize;++j)
       {
         trainingData[i][j] = vvFeature[i][j];
       }
     }
     Mat trainingDataMat(nTrainSize, nFeatureSize, CV_32FC1, trainingData);
     
     Ptr<SVM> model = SVM::create();
     model->setType(SVM::C_SVC);
     model->setKernel(SVM::LINEAR);

     Ptr<TrainData> data = TrainData::create(trainingDataMat, ROW_SAMPLE, lablesMat);   //训练数据
     model->train(data);

     model->save("car.xml");

     Ptr<ml::SVM>svm = ml::SVM::load("car.xml");
     float test[nFeatureSize] = {0};

     for(int i = 0 ; i < nFeatureSize;++i)
     {
//       std::cout << vvFeature[34][i] << std::endl;
     }
     
     int nPosCnt = 0;
     int nNegCnt = 0;
     for(int i = 0 ; i < vvFeature.size(); ++i)
     {
       for(int j = 0 ; j < nFeatureSize;++j)
        {
           test[j] = vvFeature[i][j];
        }
        Mat sample = cv::Mat(1,nFeatureSize,CV_32FC1,test);
        float result = svm->predict(sample);   //预测分区域
        if (result == 1.0)
        {
            std::cout << "yes" << std::endl;
            nPosCnt +=1;
        }
        else if(result == -1.0)
        {
            std::cout << "no" << std::endl;
            nNegCnt +=1;
        }
     }
     std::cout << vvFeature.size() << " " << nPosCnt << " " << nNegCnt << std::endl; 
     return 0;
}

bool SetDataFromFile(const char* filedir,const int w,const int h,cv::Mat &mat)
{
  std::fstream infile;
  infile.open(filedir);
   if (infile)
   {
      for(int i=0; i< h;++i)
      {
         for(int j=0;j<w;++j)
         {
             if(!infile.eof())
             {
                infile >>mat.at<uchar>(i,j);
                int m = (int)mat.at<uchar>(i,j);
                if ( m != 0)
                {
//                   std::cout << m << std::endl;
                }
             }
          }
      }
   }
  infile.close();
  return true;
}

void SetFeature(const cv::Mat mat,std::vector<float> &vFeature)
{
  //将亮度信息加入进去

  float lumdis[mat.cols]= {0};
  Getluminance(mat,lumdis);

//  float fWHRat =  GetWHFeature(mat);

  for(int i= 0 ; i < mat.rows;++i)
     vFeature.push_back(lumdis[i]);
  //将宽高比放到特征中去
//  vFeature.push_back(fWHRat);
  //将Hog特征加入进去
  cv::Mat dst;
  cv::resize(mat,dst,cv::Size(64,64));
  std::vector<float>vHogFeature(0);
  GetHogFeature(dst,vHogFeature);
  for(int i = 0 ; i < vHogFeature.size();++i)
    vFeature.push_back(vHogFeature[i]);


//  std::cout << vHogFeature.size() << std::endl;
//  for(int i = 0 ; i < vHogFeature.size();++i)
//    std::cout << vHogFeature[i] << std::endl;

}

void GetDataFromFile(const char* fileDir, const std::string path,  std::vector<std::string>& vfileDir, std::string suffix="")
{
   std::fstream infile;
   infile.open(fileDir);
   std::string fileName;
   std::string filepath = "";
   if (infile)
   {
     while(!infile.eof())
     {
       infile >>fileName;
       filepath = path+fileName + suffix;
       vfileDir.push_back(filepath);
     }
   }
   if (vfileDir.size() >0)
  	vfileDir.pop_back();
}


void GetDataFromFile(std::string fileDir, cv::Mat &m)
{
   cv::Mat matsrc = cv::imread(fileDir, cv::IMREAD_GRAYSCALE);
   cv::resize(matsrc, m, cv::Size(512, 512));
   int nRows = m.rows;
   int nCols = m.cols;

   double maxV =0,minV = 0;
   Point maxP,minP;
   minMaxLoc(m,&minV,&maxV,&minP,&maxP);
   for (int i = 0 ; i < nRows; ++i)
   {
     for(int j = 0; j < nCols;++j)
       m.at<uchar>(i,j) = (uchar)((unsigned short)m.at<uchar>(i,j)-minV)/(maxV - minV);
   }
}



//从文件夹获取文件名，读取数据，然后归一化到[0,255]
//1:将文件存储到txt
//2:从文件读取到数据到img中
void getData(const char* fileDir, const std::string path, const int w, const int h, std::vector< std::vector<float> > & vvFeature)
{
   int nIndex = 0;
   std::vector<float> vFeature(0);
   std::vector< std::string> vfileDir(0);
   GetDataFromFile(fileDir, path, vfileDir); 
   cv::Mat mat(w,h,CV_8UC1);
   while(nIndex < vfileDir.size())
   {
       if (nIndex >= 0)
       {
          const char* fileDir = &vfileDir[nIndex][0];
          GetDataFromFile(fileDir, mat);
          SetFeature(mat,vFeature);
          vvFeature.push_back(vFeature);
       }
       nIndex = nIndex + 1;
       std::cout << nIndex << std::endl;
   }
}

//连续读取文件夹下的数据
int getData(const char* file1,const char* file2,const int w,const int h,std::vector< std::vector<float> > &vvFeature)
{

  //vector:用于存储得到的一张图片的特征信息
  std::vector<float> vFeature(0);
  DIR *dp;
  struct dirent *dirp;
  int n=0;
  char c[100];
  std::string s = file1;

  if((dp=opendir(file1))==NULL)
  {
    printf("can't open file");
  }

  float lumdis[w]= {0};
  cv::Mat mat(w,h,CV_8UC1);
  int t = 0; 
  while (((dirp=readdir(dp))!=NULL) && t < 100)
  {
     n++;
     std::string filename = dirp->d_name;
     if(filename.find("txt") < 100)
     {
       std::string dir = s + filename;
       strcpy(c,dir.c_str());
//       std::cout << c << std::endl;
       SetDataFromFile(c,w,h,mat);

      for(int i = 0; i < mat.rows;++i)
      {
        for(int j = 0 ; j < mat.cols;++j)
        {
//         std::cout << (int)mat.at<uchar>(i,j) << std::endl;
        }
      }
      SetFeature(mat,vFeature);
      vvFeature.push_back(vFeature);
      vFeature.clear();
     }
     t++;
  } 
  // 导入第二批数据  
   s = file2; 
   if((dp=opendir(file2))==NULL)
   {
      printf("can't open file2");
      return 0;
   } 
   while (((dirp=readdir(dp))!=NULL)  && t < 100)
   {
       n++;
       std::string filename = dirp->d_name;
       if(filename.find("txt") < 100)
       {
         std::string dir = s + filename;
         strcpy(c,dir.c_str());
//         std::cout << c << std::endl;
         SetDataFromFile(c,w,h,mat);
            
         for(int i = 0; i < mat.rows;++i)
         {
           for(int j = 0 ; j < mat.cols;++j)
           {
//              std::cout << (int)mat.at<uchar>(i,j) << std::endl;
           }
         }
         SetFeature(mat,vFeature);
         vvFeature.push_back(vFeature);
         vFeature.clear();
     }
     t++;
  }
   if(vvFeature.size() > 2)
   {
     for(int i = 0 ; i < vvFeature[0].size();++i)
     {
//       std::cout << vvFeature[2][i] << std::endl;
     }
   }
  closedir(dp);
  return 0;
}

//获取在y轴方向上的亮度点
void Getluminance(const cv::Mat mat,float lumdis[])
{
  int nHeight = mat.rows;
  int nWidth  = mat.cols;
//  cv::Mat colvec(1,nWidth,CV_64FC1,Scalar(0));
  cv::Mat colvec;
  cv::reduce(mat,colvec,0,REDUCE_SUM,CV_32F);
 
  
  for(int i = 0; i < nWidth;++i)
  {
    for(int j = 0; j < nWidth;++j)
    {
//      std::cout <<  (int)mat.at<uchar>(i,j) << std::endl;
    }
  }  
 
  cv::Mat m1(1,nWidth,CV_64FC1,Scalar(0));
  for(int i = 0 ; i < nWidth;++i)
  {
     m1.at<double>(0,i) = (double)colvec.at<uchar>(0,i);
  }  

//  std::cout << colvec.rows << " " << colvec.cols << std::endl;
//  std::cout << (int)colvec.at<uchar>(0,10) << std::endl;
 
  cv::Mat tmp_m, tmp_sd;
  double m = 0, sd = 0; 
  cv::meanStdDev(colvec,tmp_m,tmp_sd);
  m = tmp_m.at<double>(0,0);
  sd = tmp_sd.at<double>(0,0);
  cv::normalize(m1,m1);
  for(int i = 0;i < nWidth;++i)
  {
     lumdis[i] = m1.at<double>(0,i);
  }
}

//获取胸腔的长宽比
// 1:获取胸腔的长，利用sobel算子求
void GetTwoValueImg(const cv::Mat srcMat,cv::Mat &dstMat)
{  
   Canny(srcMat,dstMat,20,80,3);
}

//获取宽的中间值
float GetMediaW(const cv::Mat mat)
{
  std::vector<int> vLength(0);
  std::vector<int> vPixel(0);
  for(int i = 0 ; i < mat.rows;++i)
  { 
     for(int j = 0; j < mat.cols;++j)
     {
       if((int)(mat.at<uchar>(i,j)) ==255)
          vPixel.push_back(j);
     }
     if(vPixel.size() > 30)
       vLength.push_back(vPixel[vPixel.size()-5] - vPixel[5]);
     vPixel.clear();
  }
  std::sort(vLength.begin(),vLength.end());
  return vLength[vLength.size()/2]; 

}

//获取高的最大值
float GetMaxH(const cv::Mat mat)
{
   std::vector<int> vLength(0);
   std::vector<int> vPixel(0);
   for(int i = 0; i< mat.cols;++i)
   {
     for(int j = 0; j < mat.rows;++j)
     {
       if((int)(mat.at<uchar>(j,i)) ==255)
          vPixel.push_back(j);
     }
     if(vPixel.size() > 30)
       vLength.push_back(vPixel[vPixel.size()-5] - vPixel[5]);
     vPixel.clear();
   }  
  std::sort(vLength.begin(),vLength.end());
  return vLength[vLength.size()-1];
}

//获取宽高比
float GetWHRatio(const float w,const float h,const float fRat =1)
{
   return w/h * fRat;
}

float GetWHFeature(const cv::Mat mat)
{
  float fRat = 0;
  cv::Mat binMat;
  GetTwoValueImg(mat,binMat);
  float fLungW = GetMediaW(binMat);
  float fLungH = GetMaxH(binMat);
  fRat = GetWHRatio(fLungW,fLungH);
  return fRat; 
}


//获取HogFeature
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

//test:查看cnny生成的轮廓
void Test_drawImg(const cv::Mat c)
{
   imwrite("color.png", c);
}

int main()
{ 
  std::string pathsrc = "/data/TB/TBdata/training/ori/png/";
  std::string pathsoft = "/data/qian/SVMSoftData/";
  const char* filesrcDir = "/home/dongdong/qian/srctestimg.txt";
  const char* filesoftDir = "/home/dongdong/qian/softtestimg.txt";	
  std::vector< std::vector<float> > vvFeature(0);
  int nCnt = 3791;
  int w = 512, h = 512;
  getData(filesrcDir, pathsrc,w, h,vvFeature);
  getData(filesoftDir, pathsoft,w, h, vvFeature);

  // 数组的初始化，是如何的
  int lables[nCnt*2] = {0};

  for(int i = 0 ; i < nCnt;++i)
    lables[i] = 1;
  for(int i = nCnt; i < 2*nCnt;++i)
    lables[i] = -1;

  SVM_Classfy(vvFeature,lables);  
  vvFeature.clear(); 
  std::cout << "AAAAAAAAAAAAAAAAAAAAAAAAA" << std::endl; 
  return 0;
}
