//
// Created by bob on 19-3-31.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;


void feature_extraction(cv::Mat& img_1,cv::Mat& img_2,vector<cv::KeyPoint>& kp_1,vector<cv::KeyPoint>& kp_2,vector<cv::DMatch>& matches);
cv::Point2d pixel2cam(const cv::Point2f& p,cv::Mat& K);

int main(int argc,char** argv){
    if (argc != 4)
    {
        cout << "请确保传入三张图像来做特征提取与匹配demo:feature_extraction img1_path img2_path depth_image_path" << endl;
        return 1;
    }

    cv::Mat img_1 = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2],CV_LOAD_IMAGE_COLOR);
    //深度图
    cv::Mat d1 = cv::imread(argv[3],CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat K = (cv::Mat_<double>(3,3) << 520.9,0,325.1,0,521.0,249.7,0,0,1);

    vector<cv::KeyPoint> kp_1,kp_2;
    vector<cv::DMatch> matches;

    feature_extraction(img_1,img_2,kp_1,kp_2,matches);

    vector<cv::Point3f> points_3d;
    vector<cv::Point2f> points_2d;
    for (auto m : matches){
        //获取相应特征点对应的深度
        ushort d = d1.ptr<unsigned short>(int(kp_1[m.queryIdx].pt.y))[int(kp_1[m.queryIdx].pt.x)];
        if (d == 0)
            continue;

        float dd = d / 1000.0;
        cv::Point2d p1 = pixel2cam(kp_1[m.queryIdx].pt,K);
        points_2d.push_back(kp_2[m.trainIdx].pt);
        points_3d.emplace_back(p1.x*dd,p1.y*dd,dd);
    }

    cout << "3D-2D匹配点对数：" << points_3d.size() << endl;

    cv::Mat r,t;

    cv::solvePnP(points_3d,points_2d,K,cv::Mat(),r,t,false,cv::SOLVEPNP_EPNP);

    cv::Mat R;
    cv::Rodrigues(r,R);


    cout << "EPnP之后的R矩阵：" << endl << R << endl;
    cout << "EPnP估测的t：" << endl << t << endl;
}