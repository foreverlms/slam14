#ifndef CAMERA_H
#define CAMERA_H

#include "common_include.h"
#include "myslam/camera.h"
#include "myslam/config.h"

/**
 *小孔成像相机模型
 */
namespace myslam
{
class Camera
{
public:
    typedef std::shared_ptr<Camera> Ptr;
    //定义相机参数及尺度
    float fx, fy, cx, cy, depth_scale;

    Vector3d world2camera(const Vector3d &p_w, const SE3 &Tcw);
    Vector3d camera2world(const Vector3d &p_c, const SE3 &Tcw);
    Vector2d camera2pixel(const Vector3d &p_c);
    Vector3d pixel2camera(const Vector2d &p_p, double depth = 1);
    Vector3d pixel2world(const Vector2d &p_p, const SE3 &Tcw, double depth = 1);
    Vector2d world2pixel(const Vector3d &p_w, const SE3 &Tcw);
    Camera(float fx_, float fy_, float cx_, float cy_, float depth_scale_)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_), depth_scale(depth_scale_) {}

    Camera()
    {
        fx = Config::get<float>("camera.fx");
        fy = Config::get<float>("camera.fy");
        cx = Config::get<float>("camera.cx");
        cy = Config::get<float>("camera.cy");
        depth_scale = Config::get<float>("camera.depth_scale");
    }
};
} // namespace myslam

#endif // !CAMERA_H
