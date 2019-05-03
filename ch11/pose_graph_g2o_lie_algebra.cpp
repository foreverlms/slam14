#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

#include <sophus/se3.h>
#include <sophus/so3.h>

using Sophus::SE3;
using Sophus::SO3;

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
Matrix6d JRInv(SE3 e)
{
    Matrix6d J;

    J.block(0, 0, 3, 3) = SO3::hat(e.so3().log());
    J.block(0, 3, 3, 3) = SO3::hat(e.translation());
    J.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero();
    J.block(3, 3, 3, 3) = SO3::hat(e.so3().log());

    //式 11.10
    J = J * 0.5 + Matrix6d::Identity();
    return J;
}

typedef Eigen::Matrix<double, 6, 1> Vector6d;
class VertexSE3LieAlgebra : public g2o::BaseVertex<6, SE3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool read(std::istream &is)
    {
        double data[7];
        for (size_t i = 0; i < 7; i++)
        {
            is >> data[i];
        }
        setEstimate(SE3(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                        Eigen::Vector3d(data[0], data[1], data[2])));
    }

    bool write(std::ostream &os) const
    {
        os << id() << " ";
        Eigen::Quaterniond q = _estimate.unit_quaternion();
        os << _estimate.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << std::endl;

        return true;
    }

    virtual void setToOriginImpl()
    {
        _estimate = Sophus::SE3();
    }

    //左乘增量更新
    virtual void oplusImpl(const double *update)
    {
        Sophus::SE3 up(
            Sophus::SO3(update[3], update[4], update[5]),
            Eigen::Vector3d(update[0], update[1], update[2]));
        _estimate = up * _estimate;
    }
};

//边的李代数表示
class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, SE3, VertexSE3LieAlgebra, VertexSE3LieAlgebra>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool read(std::istream &is)
    {
        double data[7];
        for (size_t i = 0; i < 7; i++)
        {
            is >> data[i];
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        setMeasurement(
            Sophus::SE3(q, Eigen::Vector3d(data[0], data[1], data[2])));
        for (size_t i = 0; i < information().rows() && is.good(); i++)
        {
            for (size_t j = i; j < information().cols(); j++)
            {
                is >> information()(i, j);
                if (i != j)
                {
                    information()(j, i) = information()(i, j);
                }
            }
        }
        return true;
    }

    bool write(std::ostream &os) const
    {
        VertexSE3LieAlgebra *v1 = static_cast<VertexSE3LieAlgebra *>(_vertices[0]);
        VertexSE3LieAlgebra *v2 = static_cast<VertexSE3LieAlgebra *>(_vertices[1]);

        os << v1->id() << " " << v2->id() << " ";
        SE3 m = _measurement;
        Eigen::Quaterniond q = m.unit_quaternion();
        os << m.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        for (size_t i = 0; i < information().rows(); i++)
        {
            for (size_t j = 0; j < information().cols(); j++)
            {
                os << information()(i, j) << " ";
            }
            os << std::endl;
        }
        return true;
    }

    virtual void computeError()
    {
        Sophus::SE3 v1 = (static_cast<VertexSE3LieAlgebra *>(_vertices[0]))->estimate();
        Sophus::SE3 v2 = (static_cast<VertexSE3LieAlgebra *>(_vertices[1]))->estimate();
        //误差:eij = ln(Tij^-1 * Ti ^-1 * Tj)V;
        //式11.3
        _error = (_measurement.inverse() * v1.inverse() * v2).log();
    }

    virtual void linearizeOplus()
    {
        Sophus::SE3 v1 = (static_cast<VertexSE3LieAlgebra*>(_vertices[0]))->estimate();
        Sophus::SE3 v2 = (static_cast<VertexSE3LieAlgebra*>(_vertices[1]))->estimate();
        Matrix6d J = JRInv(SE3::exp(_error));
        //式11.8 11.9
        _jacobianOplusXi=-J*v2.inverse().Adj();
        _jacobianOplusXj=J*v2.inverse().Adj();
    }
};
int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "需要指定sphere.g2o的路径!" << std::endl;
        return 1;
    }

    std::ifstream fin(argv[1]);
    if (!fin)
    {
        std::cout << "文件不存在!" << std::endl;
        return 1;
    }

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> Block;
    Block::LinearSolverType *linearSolver = new g2o::LinearSolverCholmod<Block::PoseMatrixType>();
    Block *solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(linearSolver));

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    int vertexCnt = 0, edgeCnt = 0;

    std::vector<VertexSE3LieAlgebra *> vertices;
    std::vector<EdgeSE3LieAlgebra *> edges;


    while (!fin.eof())
    {
        std::string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT")
        {
            VertexSE3LieAlgebra *v = new VertexSE3LieAlgebra();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            vertices.push_back(v);
            if (index == 0)
            {
                v->setFixed(true);
            }
        }
        else if (name == "EDGE_SE3:QUAT")
        {
            EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
            int id1, id2;
            fin >> id1 >> id2;
            e->setId(edgeCnt++);
            //TODO: 这里和源程序不一样
            // e->setVertex(0,optimizer.vertices()[id1]);
            // e->setVertex(1,optimizer.vertices()[id2]);
            e->setVertex(0, optimizer.vertex(id1));
            e->setVertex(1, optimizer.vertex(id2));

            e->read(fin);

            optimizer.addEdge(e);
            edges.push_back(e);
        }

        if (!fin.good())
            break;
    }

    std::cout << "总共读取了" << vertexCnt << "个顶点," << edgeCnt << "条边." << std::endl;

    std::cout << "开始优化..." << std::endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    std::cout << "结束优化" << std::endl;

    std::ofstream fout("result_lie.g2o");
    for (VertexSE3LieAlgebra *v : vertices)
    {
        fout << "VERTEX_SE3:QUAT ";
        v->write(fout);
    }
    for (EdgeSE3LieAlgebra *e : edges)
    {
        fout << "EDGE_SE3:QUAT ";
        e->write(fout);
    }
    fout.close();

    return 0;
}
