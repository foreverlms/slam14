#include <Eigen/StdVector>
#include <Eigen/Core>

#include <iostream>
#include <stdint.h>

#include <unordered_set>
#include <memory>
#include <vector>
#include <stdlib.h>

#include "g2o/stuff/sampler.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"

#include "../../common/BundleParams.h"
#include "../../common/BALProblem.h"
#include "g2o_bal_class.h"

using namespace Eigen;
using namespace std;

typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;
typedef g2o::BlockSolver<g2o::BlockSolverTraits<9,3>> BalBlockSolver;

void BuildProblem(const BALProblem* bal_problem,g2o::SparseOptimizer* optimizer,const BundleParams& params){
    const int num_points = bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();

    const double* raw_camera = bal_problem->cameras();
    for (size_t i = 0; i < num_cameras; i++)
    {
        //将数组转为Eigen里的Vector,长度为camera_block_size
        ConstVectorRef temVecCamera(raw_camera+camera_block_size*i,camera_block_size);
        VertexCameraBAL* pCamera = new VertexCameraBAL();
        //从文件里读入的相机估计初值
        pCamera->setEstimate(temVecCamera);
        pCamera->setId(i);
        optimizer->addVertex(pCamera);
    }

    
    const double* raw_points = bal_problem->points();
    for (size_t i = 0; i < num_points; i++)
    {
        ConstVectorRef temVecPoint(raw_points+point_block_size*i,point_block_size);
        VertexPointBAL* pPoint = new VertexPointBAL();
        pPoint->setEstimate(temVecPoint);
        pPoint->setId(i+num_cameras);

        //Schur消元
        pPoint->setMarginalized(true);
        optimizer->addVertex(pPoint);
    }
    
    const int num_observations = bal_problem->num_observations();
    const double* observations = bal_problem->observations();

    for (size_t i = 0; i < num_observations; i++)
    {
        EdgeObservationBAL* bal_edge = new EdgeObservationBAL();
        const int camera_id = bal_problem->camera_index()[i];
        const int point_id = bal_problem->point_index()[i]+num_cameras;
        if (params.robustify)
        {
            //鲁棒核函数 Huber类型
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            rk->setDelta(1.0);
            bal_edge->setRobustKernel(rk);
        }

        bal_edge->setVertex(0,dynamic_cast<VertexCameraBAL*>(optimizer->vertex(camera_id)));
        bal_edge->setVertex(1,dynamic_cast<VertexPointBAL*>(optimizer->vertex(point_id)));

        bal_edge->setInformation(Eigen::Matrix2d::Identity());

        //设定观测值,就是像素点的坐标
        bal_edge->setMeasurement(Eigen::Vector2d(observations[2*i+0],observations[2*i+1]));
        
        optimizer->addEdge(bal_edge);
    }
}

void SetSolverOptionsFromFlags(BALProblem* bal_problem,const BundleParams& params,g2o::SparseOptimizer* optimizer){
    BalBlockSolver* solver_ptr;
    g2o::LinearSolver<BalBlockSolver::PoseMatrixType>* linearSolver = 0;

    if (params.linear_solver == "dense_schur")
    {
        linearSolver = new g2o::LinearSolverDense<BalBlockSolver::PoseMatrixType>();
    }else if(params.linear_solver == "sparse_schur"){
        linearSolver = new g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>();
        //solver对矩阵进行排序以保持稀疏性
        dynamic_cast<g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>*>(linearSolver)->setBlockOrdering(true);
    }
    solver_ptr = new BalBlockSolver(unique_ptr<g2o::LinearSolver<BalBlockSolver::PoseMatrixType>>(linearSolver));

    g2o::OptimizationAlgorithmWithHessian* solver;
    if (params.trust_region_strategy == "levenberg_marquardt")
    {
        solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<BalBlockSolver>(solver_ptr));
    }else if(params.trust_region_strategy == "dogleg"){
        solver = new g2o::OptimizationAlgorithmDogleg(unique_ptr<BalBlockSolver>(solver_ptr));
    }
    else{
        std::cout << "请检查trust_region_strategy参数." << std::endl;
        exit(EXIT_FAILURE);
    }

    optimizer->setAlgorithm(solver);
}

void WriteToBALProblem(BALProblem *bal_problem, g2o::SparseOptimizer *optimizer)
{
    const int num_points = bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();

    double *raw_cameras = bal_problem->mutable_cameras();
    for (int i = 0; i < num_cameras; ++i)
    {
        VertexCameraBAL *pCamera = dynamic_cast<VertexCameraBAL *>(optimizer->vertex(i));
        Eigen::VectorXd NewCameraVec = pCamera->estimate();
        memcpy(raw_cameras + i * camera_block_size, NewCameraVec.data(), sizeof(double) * camera_block_size);
    }

    double *raw_points = bal_problem->mutable_points();
    for (int j = 0; j < num_points; ++j)
    {
        VertexPointBAL *pPoint = dynamic_cast<VertexPointBAL *>(optimizer->vertex(j + num_cameras));
        Eigen::Vector3d NewPointVec = pPoint->estimate();
        memcpy(raw_points + j * point_block_size, NewPointVec.data(), sizeof(double) * point_block_size);
    }
}

void SolveProblem(const char *filename, const BundleParams &params){
    BALProblem bal_problem(filename);

    std::cout << "BA问题文件加载..." << std::endl;
    std::cout << "BA问题有" << bal_problem.num_cameras() << " 个相机和"
              << bal_problem.num_points() << "个点." << std::endl;
    std::cout << "观测值有" << bal_problem.num_observations() << "个." << std::endl;

    // store the initial 3D cloud points and camera pose..
    if (!params.initial_ply.empty())
    {
        bal_problem.WriteToPLYFile(params.initial_ply);
    }

    std::cout << "开始求解问题..." << std::endl;

    // add some noise for the intial value
    srand(params.random_seed);
    bal_problem.Normalize();
    bal_problem.Perturb(params.rotation_sigma, params.translation_sigma,
                        params.point_sigma);
    std::cout << "噪音混淆完毕..." << std::endl;
    
    g2o::SparseOptimizer optimizer;
    SetSolverOptionsFromFlags(&bal_problem, params, &optimizer);
    BuildProblem(&bal_problem,&optimizer,params);
   
    std::cout << "开始进行图优化..." << std::endl;
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    std::cout << "结束图优化." << std::endl;

    WriteToBALProblem(&bal_problem,&optimizer);
    if (!params.final_ply.empty())
    {
        bal_problem.WriteToPLYFile(params.final_ply);

    }
    
}

int main(int argc,char** argv){
    BundleParams params(argc,argv);

    if(params.input.empty()){
        std::cout << "请确保数据集路径正确!" << std::endl;
        return 1;
    }

    SolveProblem(params.input.c_str(),params);

    return 0;
}