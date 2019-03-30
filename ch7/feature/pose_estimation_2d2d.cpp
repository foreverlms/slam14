//
// Created by bob on 19-3-28.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;

void pose_estimation_2d2d(vector<cv::KeyPoint>& kp_1,vector<cv::KeyPoint>& kp_2,vector<cv::DMatch>& matches,cv::Mat& R,cv::Mat& t){
    //相机的内参设定，这里不能用我自己的照片了，因为我不知道我这相机的内参
    cv::Mat K = (cv::Mat_<double> (3,3) << 520.9,0,325.1,0,521.0,249.7,0,0,1);

    vector<cv::Point2f> p1;
    vector<cv::Point2f> p2;

    for (int i = 0; i < matches.size(); ++i) {
        p1.push_back(kp_1[matches[i].queryIdx].pt);
        p2.push_back(kp_2[matches[i].queryIdx].pt);
    }
    //基本矩阵F=K.transpose()*E*K.inverse();
    cv::Mat fundamental_matrix;

    fundamental_matrix = cv::findFundamentalMat(p1,p2,cv::FM_8POINT);
    cout << "基本矩阵F是：" << endl;
    cout << fundamental_matrix << endl;

    cv::Point2d principal_point(325.1,249.7);//光心 TUM DATASET 标定值?
    int focal_length = 521;//焦距 TUM DATASET 标定值?

    //本质矩阵E=t^ * R
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(p1,p2,focal_length,principal_point,cv::RANSAC);
    cout << "本质矩阵是：" << endl;
    cout << essential_matrix << endl;

    //计算单应矩阵
    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(p1,p2,cv::RANSAC,3,cv::noArray(),2000,0.99);
    cout << "单应矩阵是：" << endl;
    cout << homography_matrix << endl;

    //调用opencv的从本质矩阵恢复旋转和平移信息的函数
    cv::recoverPose(essential_matrix,p1,p2,R,t,focal_length,principal_point);
    cout << "从本质矩阵恢复出的R为：" << endl;
    cout << R << endl;
    cout << "从本质矩阵恢复出的t为：" << endl;
    cout << t << endl;
}

void feature_extraction(cv::Mat& img_1,cv::Mat& img_2,vector<cv::KeyPoint>& kp_1,vector<cv::KeyPoint>& kp_2,vector<cv::DMatch>& matches){

    cv::Mat descriptors_1,descriptors_2;

    cv::Ptr<cv::ORB> orb = cv::ORB::create(500,1.2f,8,31,0,2,cv::ORB::HARRIS_SCORE,31,20);
    //通过ORB算法检测Oriented FAST角点
    orb->detect(img_1,kp_1);
    orb->detect(img_2,kp_2);

    //计算出相应角点的描述子
    orb->compute(img_1,kp_1,descriptors_1);
    orb->compute(img_2,kp_2,descriptors_2);

    cv::BFMatcher matcher;
    matcher.match(descriptors_1,descriptors_2,matches);
}

cv::Point2f pixel2cam(cv::Point2f& p,cv::Mat& K){
    return cv::Point2d( (p.x - K.at<double>(0,2)) / K.at<double>(0,1),
                        (p.y - K.at<double>(1,2)) / K.at<double>(1,1)
            );
}

int main(int argc, char** argv){
    if (argc != 3)
    {
        cout << "请确保传入两张图像来做特征提取与匹配demo:feature_extraction img1_path img2_path" << endl;
        return 1;
    }

    cv::Mat img_1 = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2],CV_LOAD_IMAGE_COLOR);

    vector<cv::KeyPoint> kp_1,kp_2;
    vector<cv::DMatch> matches;
    cout << "matches.size:" << matches.size() << endl;
    feature_extraction(img_1,img_2,kp_1,kp_2,matches);

    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    cv::Mat R ,t;

    pose_estimation_2d2d(kp_1,kp_2,matches,R,t);

    //验证E = t^R*scale;

    //t_x即是t取反对称之后的矩阵
    cv::Mat t_x = (cv::Mat_<double>(3,3) <<
            0,-t.at<double>(2,0),t.at<double>(1,0),
            t.at<double>(2,0),0,-t.at<double>(0,0),
            -t.at<double>(1,0),t.at<double>(0,0),0);
    cout << "t^R=" << endl;
    cout << t_x * R << endl;

    cv::Mat K = (cv::Mat_<double> (3,3) << 520.9,0,325.1,0,521.0,249.7,0,0,1);
    for (cv::DMatch m : matches){
        cv::Point2d pt1 = pixel2cam(kp_1[1].pt,K);
        cv::Mat y1 = (cv::Mat_<double>(3,1) << pt1.x,pt1.y,1);
        cv::Point2d pt2 = pixel2cam(kp_2[m.trainIdx].pt,K);
        cv::Mat y2 = (cv::Mat_<double>(3,1) << pt2.x,pt2.y,1);

        cv::Mat d = y2.t() * t_x * R * y1;
        cout << "对极约束为："<< endl << d << endl;
    }

    return 0;

}