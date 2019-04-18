#include "myslam/g2o_types.h"

namespace myslam
{
void EdgeProjectXYZ2UVPoseOnly::computeError()
{
    const g2o::VertexSE3Expmap *pose = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
    //重投影的误差
    //_measurement是一个uv坐标
    _error = _measurement - camera->camera2pixel(pose->estimate().map(point));
}

void myslam::EdgeProjectXYZ2UVPoseOnly::linearizeOplus()
{
    g2o::VertexSE3Expmap *pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);

    g2o::SE3Quat T(pose->estimate());
    Vector3d xyz_trans = T.map(point);
    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];
    double z_2 = z * z;

    _jacobianOplusXi(0, 0) = camera->fx * x * y / z_2;
    _jacobianOplusXi(0, 1) = -camera->fx - camera->fx * x * x / z_2;
    _jacobianOplusXi(0, 2) = camera->fx * y / z;
    _jacobianOplusXi(0, 3) = -1. / z * camera->fx;
    _jacobianOplusXi(0, 4) = 0;
    _jacobianOplusXi(0, 5) = camera->fx * x / z_2;

    _jacobianOplusXi(1, 0) = camera->fy + camera->fy * y * y / z_2;
    _jacobianOplusXi(1, 1) = -camera->fy * x * y / z_2;
    _jacobianOplusXi(1, 2) = -camera->fy * x / z;
    _jacobianOplusXi(1, 3) = 0;
    _jacobianOplusXi(1, 4) = -camera->fy / z;
    _jacobianOplusXi(1, 5) = camera->fy * y / z_2;
}
} // namespace myslam
