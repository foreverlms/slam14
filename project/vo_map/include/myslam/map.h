#ifndef MY_SLAM_MAP_H
#define MY_SLAM_MAP_H

#include <myslam/common_include.h>
#include <myslam/mappoint.h>
#include <myslam/frame.h>

namespace myslam
{
class Map
{
public:
    typedef shared_ptr<Map> Ptr;
    //地图所有的路标点
    unordered_map<unsigned long, MapPoint::Ptr> map_points;

    //地图所用到的关键帧
    unordered_map<unsigned long, Frame::Ptr> key_frames;
    Map() {}

    void insertKeyFrame(Frame::Ptr frame);
    void insertMapPoint(MapPoint::Ptr map_ppoint);
};
} // namespace myslam

#endif // !MY_SLAM_MAP_H

