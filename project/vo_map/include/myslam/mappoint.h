#ifndef MY_SLAM_MAPPOINT_H
#define MY_SLAM_MAPPOINT_H

#include <myslam/common_include.h>
#include <list>

namespace myslam
{
/**
 *前向声明，这里因为只需要myslam::Frame这个类型，并不会用到Frame是如何实现的，因此可以只声明＇\
 *Frame在这里是＂不完整类型＂
 *而且因为mappoint.cpp中只是使用了Frame的指针，并没有涉及具体的操作，因此cpp文件里也没必要包含
 *myslam/frame.h．如果要涉及具体操作，那么就要include frame.h了．
 */
class Frame;
class MapPoint
{
public:
    typedef shared_ptr<MapPoint> Ptr;
    unsigned long id;
    static unsigned long factory_id;
    bool good;
    Vector3d pos;
    //归一化？
    Vector3d norm;
    Mat descriptor;
    int visible_times;
    int matched_times;

    list<Frame *> observed_frames;

    MapPoint();
    MapPoint(
        long unsigned id_,
        const Vector3d &position_,
        const Vector3d &norm_,
        Frame *frame_ = nullptr,
        const Mat &descriptor_ = Mat());

    inline cv::Point3f getPositionCV() const
    {
        return cv::Point3f(pos(0), pos(1), pos(2));
    }

    static MapPoint::Ptr createMapPoint();
    static MapPoint::Ptr createMapPoint(
        const Vector3d &pos_world,
        const Vector3d &norm_,
        Frame *frame_, 
        const Mat &descriptor_);
};
} // namespace myslam

#endif // !MY_SLAM_MAPPOINT_H
