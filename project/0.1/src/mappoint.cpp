#include <myslam/mappoint.h>

namespace myslam
{
MapPoint::MapPoint(long id_, Vector3d position_, Vector3d norm_) : id(id_), pos(position_), norm(norm_)
{
    observed_times = 0;
    correct_times = 0;
}

MapPoint::Ptr MapPoint::createMapPoint()
{
    static int factory_id = 0;
    return MapPoint::Ptr(new MapPoint(factory_id++, Vector3d(0, 0, 0), Vector3d(0, 0, 0)));
}
} // namespace myslam
