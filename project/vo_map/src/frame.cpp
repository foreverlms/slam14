#include <myslam/frame.h>

namespace myslam
{
Frame::Frame(long id_, double time_stamp_, SE3 Tcw_, Camera::Ptr camera_, Mat color_, Mat depth_) : id(id_), time_stamp(time_stamp_), Tcw(Tcw_), camera(camera_), color(color_), depth(depth_)
{
}
Frame::~Frame()
{
}
Frame::Ptr Frame::createFrame()
{
    //静态局部变量，会累加
    static long factory_id = 0;
    return Frame::Ptr(new Frame(factory_id++));
}
double Frame::findDepth(const cv::KeyPoint &kp)
{
    int u_x = cvRound(kp.pt.x);
    int u_y = cvRound(kp.pt.y);

    ushort dep = depth.ptr<ushort>(u_y)[u_x];

    if (dep == 0)
    {
        int dx[4] = {-1, 0, 0, 1};
        int dy[4] = {0, 1, -1, 0};

        for (size_t i = 0; i < 4; i++)
        {
            dep = depth.ptr<ushort>(u_y + dx[i])[u_x + dy[i]];
            if (dep != 0)
            {
                //以该像素点周围四个点中深度不为０的点的深度代替
                return double(dep) / camera->depth_scale;
            }
        }
    }
    else
    {
        return double(dep) / camera->depth_scale;
    }

    return -1.0;
}

Vector3d Frame::getCameraCenter() const
{
    //得出相机中心在世界坐标系下的位置
    //(0,0,0)是相机坐标系的中心
    //中心世界坐标＝Tcw.inverse()*(0,0,0)，旋转矩阵被消掉，只剩下平移了(p_w = R * (0,0,0) + t)
    return Tcw.inverse().translation();
}
bool Frame::isInFrame(const Vector3d &p_w)
{
    Vector3d p_c = camera->world2camera(p_w, Tcw);
    //深度小于０的点不要
    if (p_c(2) < 0)
    {
        return false;
    }
    Vector2d p_p = camera->camera2pixel(p_c);

    return p_p(0) > 0 && p_p(1) > 0 && p_p(0) <= color.cols && p_p(1) <= color.rows;
}

} // namespace myslam
