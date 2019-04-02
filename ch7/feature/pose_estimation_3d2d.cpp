//
// Created by bob on 19-3-31.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <suitesparse/cs.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <chrono>

using namespace std;


void feature_extraction(cv::Mat& img_1,cv::Mat& img_2,vector<cv::KeyPoint>& kp_1,vector<cv::KeyPoint>& kp_2,vector<cv::DMatch>& matches);
cv::Point2d pixel2cam(const cv::Point2f& p,cv::Mat& K);

void bundleAdjustment(const vector<cv::Point3f> points_3d,
                      const vector<cv::Point2f> points_2d,
                      const cv::Mat& K,
                      cv::Mat& R,
                      cv::Mat& t)
{
    //位姿李代数维度为6，地标landmark维度为3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;
    Block::LinearSolverType* lineasolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
    Block* solver_ptr = new Block(unique_ptr<Block::LinearSolverType>(lineasolver));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<Block>(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    //顶点
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();//相机位姿
    Eigen::Matrix3d R_mat;
    R_mat << R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),
             R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
             R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2);

    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat,Eigen::Vector3d(t.at<double>(0,0),t.at<double>(0,1),t.at<double>(0,2))));
    optimizer.addVertex(pose);

    int index = 1;
    for (const cv::Point3f p : points_3d){
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x,p.y,p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }

    //设定相机参数
    g2o::CameraParameters* camera = new g2o::CameraParameters(K.at<double>(0,0),Eigen::Vector2d(K.at<double>(0,2),K.at<double>(1,2)),0);

    camera->setId(0);
    optimizer.addParameter(camera);


    //边
    index = 1;
    for(const cv::Point2f p : points_2d){
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        //将边与点一一对应？
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
        edge->setVertex(1,pose);
        edge->setMeasurement(Eigen::Vector2d(p.x,p.y));
        //以上面相机参数为准
        edge->setParameterId(0,0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
    }
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    //最多100次迭代
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "g2o图优化耗时：" << time_used.count() << "秒." << endl;

    cout << "经过g2o图优化之后的欧氏变换矩阵T：" << endl;
    cout << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
}
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

    bundleAdjustment(points_3d,points_2d,K,R,t);
}