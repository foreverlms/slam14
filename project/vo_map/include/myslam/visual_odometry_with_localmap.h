#include <myslam/visual_odometry.h>
#include <myslam/mappoint.h>
#include <opencv2/features2d/features2d.hpp>

namespace myslam
{
class VisualOdometryWithLocalMap : public VisualOdometry
{

public:
    cv::FlannBasedMatcher matcher_flann;
    vector<MapPoint::Ptr> match_3dpts;
    vector<int> match_2d_kp_index;
    SE3 Tcw_estimated;
    //匹配系数
    float match_ratio;
    double map_point_erase_ratio;

public:
    typedef shared_ptr<VisualOdometryWithLocalMap> Ptr;

    VisualOdometryWithLocalMap();
    ~VisualOdometryWithLocalMap();

    void addKeyFrame();
    bool addFrame(Frame::Ptr ptr);
    void optimizeMap();
    void featureMatching();
    void poseEstimationPnP();
    bool checkEstimatePose();
    bool checkKeyFrame();
    void addMapPoints();
    double getViewAngle(Frame::Ptr, MapPoint::Ptr point);
};
} // namespace myslam
