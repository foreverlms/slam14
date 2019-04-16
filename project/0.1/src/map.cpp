#include <myslam/map.h>

namespace myslam
{
void Map::insertKeyFrame(Frame::Ptr frame)
{
    std::cout << "现有" << key_frames.size() << "帧关键帧" << '\n';
    if (key_frames.find(frame->id) == key_frames.end())
    {
        key_frames.insert(make_pair(frame->id, frame));
    }
    else
    {
        key_frames[frame->id] = frame;
    }
}
void Map::insertMapPoint(MapPoint::Ptr point)
{
    std::cout << "现有" << map_points.size() << "帧路标点。" << '\n';
    if (map_points.find(point->id) == map_points.end())
    {
        map_points.insert(make_pair(point->id, point));
    }
    else
    {
        map_points[point->id] = point;
    }
}
} // namespace myslam
