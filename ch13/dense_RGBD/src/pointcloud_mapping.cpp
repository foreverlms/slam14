#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <boost/format.hpp>
#include <pcl-1.7/pcl/point_types.h>
#include <pcl-1.7/pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;

int main(int argc, char **argv)
{
    vector<cv::Mat> colorImgs, depthImgs;
    vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;

    // // vector<Eigen::Isometry3d> poses;

    ifstream fin("./pose.txt");

    if (!fin)
    {
        cerr << "请提供pose.txt文件!" << endl;
        return 1;
    }


    //获取每一张图片的相机与位姿
    for (int i = 0; i < 5; i++)
    {
        boost::format fmt("./%s/%d.%s");

        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1)); //读取灰度值图像

        double data[7] = {0};

        for (auto &d : data)
            fin >> d;

        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        //构建相机到世界坐标系的欧氏变换矩阵Twc
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }


    //确定相机内参
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;

    cout << "正在将图像转换为点云..." << endl;

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    PointCloud::Ptr pointCloud(new PointCloud);

    for (int i = 0; i < 5; i++)
    {
        PointCloud::Ptr curr(new PointCloud);
        
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];

        Eigen::Isometry3d T = poses[i];

        for (int v = 0; v < color.rows; v++)
        {
            for (int u = 0; u < color.cols; u++)
            {
                unsigned int d = depth.ptr<unsigned short>(v)[u];
                if (d == 0)
                {
                    continue;
                }
                if (d >= 7000)
                {
                    continue;
                }
                Eigen::Vector3d point;
                point[2] = (double)d / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d point2world = T * point;

                PointT p;
                p.x = point2world[0];
                p.y = point2world[1];
                p.z = point2world[2];
                p.b = color.data[v * color.step + u * color.channels()];
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.g = color.data[v * color.step + u * color.channels() + 2];
                curr->points.push_back(p);
            }
        }

        PointCloud::Ptr tmp(new PointCloud);
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
        statistical_filter.setMeanK(50);
        statistical_filter.setStddevMulThresh(1.0);
        statistical_filter.setInputCloud(curr);
        statistical_filter.filter(*tmp);
        (*pointCloud) += *tmp;
    }

    pointCloud->is_dense = false;
    cout << "点云共有" << pointCloud->size() << "个点" << endl;

    //体素滤波
    pcl::VoxelGrid<PointT> voxel_filter;
    voxel_filter.setLeafSize(0.01, 0.01, 0.01);
    PointCloud::Ptr tmp(new PointCloud);
    voxel_filter.setInputCloud(pointCloud);
    voxel_filter.filter(*tmp);
    tmp->swap(*pointCloud);

    // pointCloud->width =1;
    // pointCloud->height=pointCloud->size();
    cout << "滤波之后点云共有" << pointCloud->size() << "个点." << endl;

    pcl::RangeImage rm;
    rm.createFromPointCloud(*pointCloud, pcl::deg2rad(0.5f), pcl::deg2rad(360.0f), pcl::deg2rad(180.0f), (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f),pcl::RangeImage::CAMERA_FRAME,0.0f,0.0f,1);
    pcl::visualization::RangeImageVisualizer range_viewer("Range Image");
    range_viewer.showRangeImage(rm);
    
    cout << rm << endl;

    float* ranges = rm.getRangesArray();
    unsigned char* rgbd_image = pcl::visualization::FloatImageUtils::getVisualImage(ranges,rm.width,rm.height);
    pcl::io::saveRgbPNGFile("range_image.png",rgbd_image,rm.width,rm.height);
    pcl::io::savePCDFileBinary("map.pcd", *pointCloud);
    
    return 0;
}