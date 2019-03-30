//
// Created by bob on 19-3-30.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>

using namespace std;

//这里要把形参设置为const类型的引用
cv::Point2d pixel2cam(const cv::Point2f& p,cv::Mat& K);
void feature_extraction(cv::Mat& img_1,cv::Mat& img_2,vector<cv::KeyPoint>& kp_1,vector<cv::KeyPoint>& kp_2,vector<cv::DMatch>& matches);
void pose_estimation_2d2d(vector<cv::KeyPoint>& kp_1,vector<cv::KeyPoint>& kp_2,vector<cv::DMatch>& matches,cv::Mat& R,cv::Mat& t);

void triangulation(const vector<cv::KeyPoint>& kp_1,
                   const vector<cv::KeyPoint>& kp_2,
                   const vector<cv::DMatch>& matches,
                   const cv::Mat& R,
                   const cv::Mat& t,
                   vector<cv::Point3d>& points);

void triangulation(const vector<cv::KeyPoint>& kp_1,
                   const vector<cv::KeyPoint>& kp_2,
                   const vector<cv::DMatch>& matches,
                   const cv::Mat& R,
                   const cv::Mat& t,
                   vector<cv::Point3d>& points){
    //第一幅图片为基准，设其T为单位阵加0平移，即平移初始化
    cv::Mat T1 = (cv::Mat_<double>(3,4) << 1,0,0,0,
                                           0,1,0,0,
                                           0,0,1,0);
    cv::Mat T2 = (cv::Mat_<double>(3,4) <<
            R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),t.at<double>(0,0),
            R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),t.at<double>(1,0),
            R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2),t.at<double>(2,0));

    cv::Mat K = (cv::Mat_<double>(3,3) << 520.9,0,325.1,0,521.0,249.7,0,0,1);
    vector<cv::Point2d> p_1,p_2;

    for (cv::DMatch m : matches){
        p_1.push_back(pixel2cam(kp_1[m.queryIdx].pt,K));
        p_2.push_back(pixel2cam(kp_2[m.trainIdx].pt,K));
    }

    cv::Mat point_4d;
    //三角测量
    cv::triangulatePoints(T1,T2,p_1,p_2,point_4d);

    for (int i = 0; i < point_4d.cols; ++i) {
        cv::Mat x = point_4d.col(i);
        x /= x.at<float>(3,0);
        cv::Point3d p (x.at<float>(0,0),x.at<float>(1,0),x.at<float>(2,0));

        points.push_back(p);
    }

}

int main(int argc,char** argv){

    if (argc != 3)
    {
        cout << "请确保传入两张图像来做特征提取与匹配demo:feature_extraction img1_path img2_path" << endl;
        return 1;
    }

    cv::Mat img_1 = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2],CV_LOAD_IMAGE_COLOR);

    vector<cv::KeyPoint> kp_1,kp_2;
    vector<cv::DMatch> matches;

    //提取ORB特征点
    feature_extraction(img_1,img_2,kp_1,kp_2,matches);

    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    cv::Mat R ,t;

    //估计本质矩阵，恢复出两帧图片之间的R与t
    pose_estimation_2d2d(kp_1,kp_2,matches,R,t);

    vector<cv::Point3d> points;
    //三角测量
    triangulation(kp_1,kp_2,matches,R,t,points);

    cv::Mat K = (cv::Mat_<double> (3,3) << 520.9,0,325.1,0,521.0,249.7,0,0,1);

    for (int i = 0; i < matches.size(); ++i) {
        cv::Point2d pt1_cam = pixel2cam(kp_1[matches[i].queryIdx].pt,K);
//        cv::Point2d pt1_cam_tr (points[i].x / points[i].z,points[i].y / points[i].z);
        cv::Mat pt1_cam_tr = (cv::Mat_<double>(3,1) << points[i].x / points[i].z,points[i].y / points[i].z,1);

        cout << "**********************" << endl;
        cout << "从第一幅图里得到的特征点在归一化平面上的归一化坐标：" << pt1_cam << endl << endl;
        cout << "三角化之后取得的真实点在第一幅图中相机坐标系下的真实坐标（包含深度z）归一化之后的坐标：" << endl << pt1_cam_tr.t() <<"深度是：" << points[i].z << endl;
//        cout << "----------------------" << endl;

        cv::Point2d pt2_cam = pixel2cam(kp_2[matches[i].trainIdx].pt,K);
        //通过x2 = R * x1 + t来算出按照第一幅图片恢复出来的点坐标在第二幅图片里的归一化坐标
        cv::Mat pt2_cam_tr = R * (cv::Mat_<double>(3,1) << points[i].x ,points[i].y,points[i].z) + t;
        pt2_cam_tr /= pt2_cam_tr.at<double>(2,0);

        cout << "######################" << endl;
        cout << "从第二幅图里得到的特征点在归一化平面上的归一化坐标：" << pt2_cam << endl << endl;
        cout << "三角化之后取得的真实点在第二幅图中相机坐标系下的真实坐标归一化之后的坐标：" << endl << pt2_cam_tr.t() << endl;
        cout << "//////////////////////" << endl;
    }

    //验证三角化点与特征点的重投影关系
    return 0;
}