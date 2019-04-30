#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

using namespace std;

int main(int argc,char** argv){
    if (argc != 2)
    {
        std::cout << "需要指定sphere.g2o的路径!" << std::endl;
        return 1;
    }

    ifstream fin(argv[1]);
    if(!fin){
        std::cout << "文件不存在!" << std::endl;
        return 1;
    }

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,6>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCholmod<Block::PoseMatrixType>();
    Block* solver_ptr = new Block(unique_ptr<Block::LinearSolverType>(linearSolver));

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<Block>(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    int vertexCnt =0,edgeCnt = 0;
    while (!fin.eof()) 
    {
        string name;
        fin >> name;
        if(name == "VERTEX_SE3:QUAT"){
            g2o::VertexSE3* v = new g2o::VertexSE3();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            if(index == 0){
                v->setFixed(true);
            }
        }else if (name == "EDGE_SE3:QUAT")
        {
            g2o::EdgeSE3* e = new g2o::EdgeSE3();
            int id1,id2;
            fin >> id1 >> id2;
            e->setId(edgeCnt++);
            //TODO: 这里和源程序不一样
            // e->setVertex(0,optimizer.vertices()[id1]);
            // e->setVertex(1,optimizer.vertices()[id2]);
            e->setVertex(0, optimizer.vertex(id1));
            e->setVertex(1, optimizer.vertex(id2));

            e->read(fin);

            optimizer.addEdge(e);
        }

        if(!fin.good()) break;
    }

    cout << "总共读取了" << vertexCnt << "个顶点," << edgeCnt << "条边." << endl;

    cout << "开始优化..." << endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    cout << "结束优化" << endl;

    optimizer.save("result.g2o");

    return 0;
    
    
}