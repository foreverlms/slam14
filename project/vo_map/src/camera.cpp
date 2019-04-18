#include <myslam/camera.h>

namespace myslam
{
Vector3d Camera::world2camera(const Vector3d &p_w, const SE3 &Tcw)
{
    //Tcw是李群
    return Tcw * p_w;
}
Vector3d Camera::camera2world(const Vector3d &p_c, const SE3 &Tcw)
{
    return Tcw.inverse() * p_c;
}
Vector2d Camera::camera2pixel(const Vector3d &p_c)
{
    return Vector2d(fx * p_c(0) / p_c(2) + cx, fy * p_c(1) / p_c(2) + cy);
}
Vector3d Camera::pixel2camera(const Vector2d &p_p, double depth)
{
    return Vector3d((p_p(0) - cx) * depth / fx, (p_p(1) - cy) * depth / fy, depth);
}
Vector3d Camera::pixel2world(const Vector2d &p_p, const SE3 &Tcw, double depth)
{
    return camera2world(pixel2camera(p_p, depth), Tcw);
}
Vector2d Camera::world2pixel(const Vector3d &p_w, const SE3 &Tcw)
{
    return camera2pixel(world2camera(p_w, Tcw));
}


} // namespace myslam
