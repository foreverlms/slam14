#ifndef MY_SLAM_MAPPOINT_H
#define MY_SLAM_MAPPOINT_H

#include <myslam/common_include.h>

namespace myslam
{
class MapPoint
{
public:
    typedef shared_ptr<MapPoint> Ptr;
    unsigned long id;
    Vector3d pos;
    Vector3d norm;
    Mat descriptor;
    int observed_times;
    int correct_times;

    MapPoint() {}
    MapPoint(long id_, Vector3d position_, Vector3d norm_);

    static MapPoint::Ptr createMapPoint();
};
} // namespace myslam

#endif // !MY_SLAM_MAPPOINT_H