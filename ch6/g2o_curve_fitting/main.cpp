//
// Created by bob on 19-3-21.
//

#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;

//曲线模型的顶点，模板参数分别为：优化变量维度和数据类型
class CurveFittingVertex:public g2o::BaseVertex<3,Eigen::Vector3d>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginTmpl(){
        _estimate << 0,0,0;
    }

    virtual void oplusImpl(const double* update){
        _estimate += Eigen::Vector3d(update);
    }

    virtual bool read(istream& in){}

    virtual bool write(ostream& out){}
};

//曲线模型的边
class CurveFittingEdge:public g2o::BaseUnaryEdge<1,double ,CurveFittingVertex>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge(double x) : BaseUnaryEdge(),_x(x){}
    void computeError(){
        const auto * v = dynamic_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d& abc = v->estimate();
        _error(0,0) = _measurement - std::exp(abc(0,0) * _x * _x + abc(1,0) * _x + abc(2,0));
    }
    virtual bool read(istream& in){}
    virtual  bool write(ostream& out) const {}
public:
    double _x ;
};
int main(int argc,char** argv){
    double a = 1.0,b=2.0,c=1.0;
    int N = 100;
    double w_sigma = 1.0;
    cv::RNG rng;
    double abc[3] = {0,0,0};

    vector<double> x_data,y_data;

    cout <<"正在生成真实数据..." << endl;

    for (int i = 0; i < N; ++i) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(abc[0] * x * x + abc[1] * x +abc[2])+rng.gaussian(w_sigma));

    }

    //矩阵块
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
//    unique_ptr<Block::LinearSolverType> lin(linearSolver);
    //TODO 这里为啥一直出错？
    Block* solver_ptr = new Block(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    CurveFittingVertex* v = new CurveFittingVertex();


    return 0;
}