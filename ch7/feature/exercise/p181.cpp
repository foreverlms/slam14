//
// Created by bob on 19/04/10.
//

/**
 * 2、设计程序调用OpenCV中的其他种类特征点。统计在提取1000个特征点时在你的机器上所用的时间
 */



#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#define MAX_POINTS 1000
#define FILE_PATH "/home/bob/CLionProjects/slam14/ch7/feature/p1.jpg"
using namespace std;

cv::Mat img = cv::imread(FILE_PATH);

inline chrono::steady_clock::time_point getNow(){
    return chrono::steady_clock::now();
}

/**
 * HOG特征的提取
 */
void getSIFT(){
    vector<cv::KeyPoint> kp;
    cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create(1000);

    auto t1 = getNow();
    detector->detect(img, kp);
    auto t2 = getNow();

    std::cout << "SIFT总共找出来" << kp.size() << "个特征点" << std::endl;
    chrono::duration<double> time_used = t2-t1;
    std::cout << "耗时：" << time_used.count() << "秒" << std::endl;
    cv::Mat c_image;
    img.copyTo(c_image);

    if (!kp.empty())
    {
        for (auto k : kp)
        {
            cv::circle(c_image, k.pt, 5, cv::Scalar(0, 240, 0), 1);
        }

        cv::imshow("SIFT", c_image);
        cv::waitKey(0);
    }
}

//TODO: 做题！
/**
 * SURF特征提取
 */


void getSURF(){
    vector<cv::KeyPoint> kp;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(2800);

    auto t1 = getNow();
    detector->detect(img, kp);
    auto t2 = getNow();

    std::cout << "SURF总共找出来" << kp.size() << "个特征点" << std::endl;
    chrono::duration<double> time_used = t2 - t1;
    std::cout << "耗时：" << time_used.count() << "秒" << std::endl;
    cv::Mat c_image;
    img.copyTo(c_image);

    if (!kp.empty())
    {
       for (auto k : kp){
           cv::circle(c_image,k.pt,5,cv::Scalar(0,240,0),1);
       } 

       cv::imshow("SURF",c_image);
       cv::waitKey(0);
    }
}

void getBRISK(){
    vector<cv::KeyPoint> kp;
    cv::Ptr<cv::BRISK> detector = cv::BRISK::create();

    auto t1 = getNow();
    detector->detect(img, kp);
    auto t2 = getNow();

    std::cout << "BRISK总共找出来" << kp.size() << "个特征点" << std::endl;
    chrono::duration<double> time_used = t2 - t1;
    std::cout << "耗时：" << time_used.count() << "秒" << std::endl;
    cv::Mat c_image;
    img.copyTo(c_image);

    if (!kp.empty())
    {
        for (auto k : kp)
        {
            cv::circle(c_image, k.pt, 5, cv::Scalar(0, 240, 0), 1);
        }

        cv::imshow("BRISK", c_image);
        cv::waitKey(0);
    }
}

void getAKAZE(){
    vector<cv::KeyPoint> kp;
    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();

    auto t1 = getNow();
    detector->detect(img, kp);
    auto t2 = getNow();

    std::cout << "AKAZE总共找出来" << kp.size() << "个特征点" << std::endl;
    chrono::duration<double> time_used = t2 - t1;
    std::cout << "耗时：" << time_used.count() << "秒" << std::endl;
    cv::Mat c_image;
    img.copyTo(c_image);

    if (!kp.empty())
    {
        for (auto k : kp)
        {
            cv::circle(c_image, k.pt, 5, cv::Scalar(0, 240, 0), 1);
        }

        cv::imshow("AKAZE", c_image);
        cv::waitKey(0);
    }
}
/**
 * 输出：
 * 
 *SURF总共找出来779个特征点
**耗时：0.153737秒
**SIFT总共找出来1000个特征点
**耗时：0.487359秒
**BRISK总共找出来4331个特征点
**耗时：0.081469秒
**AKAZE总共找出来2829个特征点
**耗时：0.277592秒
 */
void show(){
    cv::imshow("show",img);
    cv::waitKey(0);
}
int main(int argc, char** argv){
    getSURF();
    getSIFT();
    getBRISK();
    getAKAZE();
}