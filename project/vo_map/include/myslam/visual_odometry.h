#ifndef MY_SLAM_VISUAL_ODOMETRY_H
#define MY_SLAM_VISUAL_ODOMETRY_H

#include <myslam/common_include.h>
#include <myslam/map.h>
#include <myslam/frame.h>
namespace myslam
{
class VisualOdometry
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState
    {
        INITIALIZING = -1,
        OK = 0,
        LOST
    };
    VOState state;
    Map::Ptr map;
    Frame::Ptr ref;
    Frame::Ptr curr;

    cv::Ptr<cv::ORB> orb;
    vector<cv::Point3f> pts_3d_ref;
    vector<cv::KeyPoint> kps_curr;
    Mat descriptors_curr, descriptors_ref;
    vector<cv::DMatch> feature_matches;

    SE3 Tcr_estimated; //估计的参考帧（前一帧）到当前帧的位姿
    int num_inliers;   //可用的点吗？
    int num_lost;

    //需要从配置文件中读取的数据
    int num_of_features;
    double scale_factor;
    int level_pyramid;
    float match_ratio;

    int max_num_lost; //记录丢失帧数
    int min_inliers;

    double key_frame_min_rot;
    double key_frame_min_trans;

public:
    VisualOdometry();
    ~VisualOdometry();

    bool addFrame(Frame::Ptr ptr);

protected:
    void extractKeyPoints();
    void computeDescriptors();
    void featureMatching();
    void poseEstimationPnP();
    void setRef3DPoints();

    void addKeyFrame();
    bool checkEstimatePose();
    bool checkKeyFrame();
};
} // namespace myslam

#endif // !MY_SLAM_VISUAL_ODOMETRY_H

