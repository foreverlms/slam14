//
// Created by bob on 19-2-28.
//

#include <iostream>
#include <ctime>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "eigenMatrix.h"

#define MATRIX_SIZE 50;

using namespace Eigen;

int matrix(){
    //2*3 matrix
    Eigen::Matrix<float,2,3> matrix_23;
    //column matrix 3*1
    Eigen::Vector3d v_3d;

    Matrix3d matrix_33 = Matrix3d::Zero();

    //dynamic matrix
    Matrix<double ,Dynamic,Dynamic> matrix_dynamic;
    MatrixXd matrix_x;

    //matrix operation
    matrix_23 << 1,2,3,4,5,6;
    std::cout << matrix_23 << std::endl;
    std::cout << "matrix_23的初始化" << std::endl;

    for (int i = 0; i <= 1; ++i) {
        for (int j = 0; j <= 2; ++j) {
            std::cout << matrix_23(i,j) << std::endl;
        }
    }

    v_3d << 1,2,3;
    Matrix<double,2,1> result = matrix_23.cast<double>() * v_3d;
    std::cout << "显式转换float->double，2*3矩阵和3*1列向量相乘" << std::endl;
    std::cout << result << std::endl;
    matrix_33 = Matrix3d::Random();

    std::cout << matrix_33 << std::endl;

    std::cout << matrix_33.sum() << std::endl;

    std::cout << matrix_33.trace() << std::endl;

    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);

    std::cout << "Eigen values(特征值) = " << std::endl << eigen_solver.eigenvalues() << std::endl;

    Eigen::Matrix<double,50,50> matrix_NN;

    //求50*50矩阵matrix_NN*x = v_Nd这个方程的解x
    //直接求逆和QR分解的性能差异
    matrix_NN = MatrixXd::Random(50,50);
    Matrix<double,50,1> v_Nd;

    v_Nd = MatrixXd::Random(50,1);

    clock_t time_start = clock();
    Matrix<double,50,1> x = matrix_NN.inverse()*v_Nd;
    clock_t time_end = clock();

    std::cout << 1000 * (time_end - time_start)/(double) CLOCKS_PER_SEC << "ms" << std::endl;

    time_start = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    std::cout << 1000*(clock() - time_start )/(double) CLOCKS_PER_SEC << "ms" << std::endl;

    //QR分解的求解速度并没有比直接求逆快多少啊
    return 0;
}