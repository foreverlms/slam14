#ifndef MY_SLAM_G2O_TYPES_H
#define MY_SLAM_G2O_TYPES_H

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "common_include.h"
#include "camera.h"
namespace myslam
{
class EdgeProjectXYZ2UVPoseOnly : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void computeError();
    virtual void linearizeOplus();
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &os) const {}

    Camera *camera;
    Vector3d point;
};
} // namespace myslam

#endif // !MY_SLAM_G2O_TYPES_H
