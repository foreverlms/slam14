#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>

#include <Eigen/Core>
#include "autodiff.h"

#include "../../common/flags/command_args.h"
#include "../../common/projection.h"

/*优化的变量包括相机六维李代数位姿以及三维的相机参数f(焦距),k1(径向畸变),k2(切向畸变)*/
class VertexCameraBAL : public g2o::BaseVertex<9, Eigen::VectorXd>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraBAL() {}
    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double *update) override
    {
        Eigen::VectorXd::ConstMapType v(update, VertexCameraBAL::Dimension);
        std::cout << "Dimension维度是:" << VertexCameraBAL::Dimension << std::endl;
        _estimate += v;
    }
};

/*路标点为三维空间点*/
class VertexPointBAL : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointBAL() {}
    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    virtual void setToOriginImpl() {}
    virtual void oplusImpl(const double *update) override
    {
        Eigen::Vector3d::ConstMapType v(update);
        _estimate += v;
    }
};

/*边,包含着重投影误差,也就是两维*/
class EdgeObservationBAL : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexCameraBAL, VertexPointBAL>
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeObservationBAL(){}

        virtual bool read(std::istream& /*is*/){return false;}
        virtual bool write(std::ostream& /*os*/) const {return false;}

        virtual void computeError() override{
            const VertexCameraBAL* cam  = static_cast<const VertexCameraBAL*>(vertex(0));
            const VertexPointBAL* point = static_cast<const VertexPointBAL*>(vertex(1));

            (*this)(cam->estimate().data(),point->estimate().data(),_error.data());
        }

        template<typename T>
        bool operator()(const T * camera,const T * point,T* residuals) const{
            T predictions[2];

            CamProjectionWithDistortion(camera,point,predictions);
            residuals[0] = predictions[0]-T(measurement()(0));
            residuals[1] = predictions[1] - T(measurement()(1));

            return true;
        }

        virtual void linearizeOplus() override{
            const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*>(vertex(0));

            const VertexPointBAL* point = static_cast<const VertexPointBAL*>(vertex(1));

            typedef ceres::internal::AutoDiff<EdgeObservationBAL,double,VertexCameraBAL::Dimension,VertexPointBAL::Dimension> BalAutoDiff;

            Eigen::Matrix<double,Dimension,VertexCameraBAL::Dimension,Eigen::RowMajor> dError_dCamera;
            Eigen::Matrix<double,Dimension,VertexPointBAL::Dimension,Eigen::RowMajor> dError_dPoint;

            double* parameters[] = {const_cast<double*>(cam->estimate().data())};
            const_cast<double*>(point->estimate().data());

            double* jacobians[] = {dError_dPoint.data(),dError_dPoint.data()};
            double value[Dimension];

            bool diffState = BalAutoDiff::Differentiate(*this,parameters,Dimension,value,jacobians);
            if(diffState){
                _jacobianOplusXi = dError_dCamera;
                _jacobianOplusXj = dError_dPoint;
            }
            else{
                std::cout << "求导出错" << std::endl;
                _jacobianOplusXi.setZero();
                _jacobianOplusXj.setZero();
            }
        }
};