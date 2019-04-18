#include <myslam/mappoint.h>
namespace myslam
{
MapPoint::MapPoint() : id(-1), pos(Vector3d(0, 0, 0)), norm(Vector3d(0, 0, 0)), good(true), visible_times(0), matched_times(0)
{
}

MapPoint::MapPoint(long unsigned id_, const Vector3d &pos_, const Vector3d &norm_, Frame *frame_, const Mat &descriptor_) : id(id_), pos(pos_), norm(norm_), good(true), visible_times(1), matched_times(1), descriptor(descriptor_)
{
    observed_frames.push_back(frame_);
}

MapPoint::Ptr MapPoint::createMapPoint()
{

    return MapPoint::Ptr(new MapPoint(factory_id++, Vector3d(0, 0, 0), Vector3d(0, 0, 0)));
}

MapPoint::Ptr MapPoint::createMapPoint(const Vector3d &pos_world,
                                       const Vector3d &norm_,
                                       Frame *frame_,
                                       const Mat &descriptor_)
{
    return MapPoint::Ptr(new MapPoint(factory_id++, pos_world, norm_,frame_, descriptor_));
}

unsigned long MapPoint::factory_id = 0;
} // namespace myslam
