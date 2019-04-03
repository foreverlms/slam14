//
// Created by bob on 19-4-3.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

void feature_extraction(cv::Mat& img_1,cv::Mat& img_2,vector<cv::KeyPoint>& kp_1,vector<cv::KeyPoint>& kp_2,vector<cv::DMatch>& matches);
cv::Point2d pixel2cam(const cv::Point2f& p,cv::Mat& K);

void pose_estimation_3d3d(const vector<cv::Point3f>& pts1, const vector<cv::Point3f>& pts2, cv::Mat& R, cv::Mat& t){
    //质心计算
    cv::Point3f p1,p2;
    int n = pts1.size();

    for (int i = 0; i < n; ++i) {
        p1 += pts1[i];
        p2 += pts2[i];
    }

    p1 /= n;
    p2 /= n;

    vector<cv::Point3f> q1(n),q2(n);
    for (int j = 0; j < n; ++j) {
        q1[j] = pts1[j] - p1;
        q2[j] = pts2[j] - p2;
    }


    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int k = 0; k < n; ++k) {
        //这里的W矩阵按P174的说法是计算qi*qi'.transpose()，需要查看文献50,51
        W += Eigen::Vector3d(q1[k].x,q1[k].y,q2[k].z) * Eigen::Vector3d(q2[k].x,q2[k].y,q2[k].z).transpose();
    }

    cout << "矩阵W是：" << endl << W;

    //W的SVD分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W,Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    cout << "W左奇异矩阵U：" << endl << U << endl;
    cout << "W右奇异矩阵V：" << endl << V << endl;

    Eigen::Matrix3d R_ = U*(V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x,p1.y,p1.z) - R_*Eigen::Vector3d(p2.x,p2.y,p2.z);



    R = (cv::Mat_<double>(3,3) << R_(0,0),R_(0,1),R_(0,2),
                                  R_(1,0),R_(1,1),R_(1,2),
                                  R_(2,0),R_(1,1),R_(2,2));

    t = (cv::Mat_<double>(3,1) << t_(0,0),t_(1,0),t_(2,0));

}
int main(int argc,char** argv){
    if (argc != 5)
    {
        cout << "请确保传入四张图像来做特征提取与匹配demo:feature_extraction img1_path img2_path depth_image1_path depth_image2_depth" << endl;
        return 1;
    }

    cv::Mat img_1 = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2],CV_LOAD_IMAGE_COLOR);
    //深度图
    cv::Mat d1 = cv::imread(argv[3],CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat d2 = cv::imread(argv[4],CV_LOAD_IMAGE_UNCHANGED);

    cv::Mat K = (cv::Mat_<double>(3,3) << 520.9,0,325.1,0,521.0,249.7,0,0,1);

    vector<cv::KeyPoint> kp_1,kp_2;
    vector<cv::DMatch> matches;

    feature_extraction(img_1,img_2,kp_1,kp_2,matches);
    cout << "提取出" << kp_1.size() << "对匹配点" << endl;

    vector<cv::Point3f> points1_3d;
    vector<cv::Point3f> points2_3d;

    vector<cv::Point2f> points_2d;
    for (auto m : matches){
        //获取相应特征点对应的深度
        ushort dep1 = d1.ptr<unsigned short>(int(kp_1[m.queryIdx].pt.y))[int(kp_1[m.queryIdx].pt.x)];
        ushort dep2 = d2.ptr<unsigned short>(int(kp_2[m.trainIdx].pt.y))[int(kp_2[m.trainIdx].pt.x)];
        if (dep1 == 0 || dep2 == 0)
            continue;
        cv::Point2d p1 = pixel2cam(kp_1[m.queryIdx].pt,K);
        cv::Point2d p2 = pixel2cam(kp_2[m.trainIdx].pt,K);

        float dd1 = float ( dep1 ) /5000.0;
        float dd2 = float ( dep2 ) /5000.0;

        points1_3d.emplace_back(cv::Point3f(p1.x,p1.y,dd1));
        points2_3d.emplace_back(cv::Point3f(p2.x,p2.y,dd2));
    }

    cout << "3D-3D匹配点对数：" << points1_3d.size() << endl;

    cv::Mat R,t;
    pose_estimation_3d3d(points1_3d,points2_3d,R,t);
    cout << "通过SVD分解的ICP求解结果：" << endl;


    cout << "R矩阵：" << endl << R << endl;
    cout << "t平移：" << endl << t << endl;
}