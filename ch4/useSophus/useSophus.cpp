//
// Created by bob on 19-3-14.
//

#include <iostream>
#include <cmath>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "Sophus/sophus/so3.h"
#include "Sophus/sophus/se3.h"


int main(int argc,char* argv[]){

    cout << "特殊正交群***************************" << endl;
    //由旋转向量构造旋转矩阵，旋转轴为z轴，角度为90度
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2,Eigen::Vector3d(0,0,1)).toRotationMatrix();
    cout << "旋转矩阵R为：\n" << R << endl;

    //特殊正交群
    Sophus::SO3 s3(R);
    cout << "R对应的SO3群是（以李代数形式输出）: \n" << s3 << endl;

    Eigen::Vector3d lie_so3 = s3.log();
    cout << "R求得的 so3 李代数（向量）：\n" << lie_so3.transpose() << endl;

    cout << "李代数lie_so3对应的反对称矩阵是：" << endl;
    cout << Sophus::SO3::hat(lie_so3) << endl;

    Eigen::Vector3d update_so3(1e-4,0,0);
    Sophus::SO3 so3_updated = Sophus::SO3::exp(update_so3) * s3;
    cout << "左扰动模型（1e-4,0,0）：\n" <<  so3_updated << endl;

    cout << "特殊欧式群****************************" << endl;
    //SE(3)特殊欧式群
    Eigen::Vector3d t(1,0,0);
    Sophus::SE3 se3(R,t);
    cout << "由旋转矩阵和平移向量来构造se3：\n" << se3.matrix() << endl;

    typedef Eigen::Matrix<double ,6,1> Vector6d;

    Vector6d lie_se3 = se3.log();
    cout << "se3对应的李代数（六维向量形式表示，前三行为旋转，后三行为平移）：\n" << lie_se3.transpose() << endl;

    cout << "李代数lie_se3对应的反对称矩阵为：\n" << Sophus::SE3::hat(lie_se3) << endl;

    cout << "先反对称hat在到向量vee：\n" << Sophus::SE3::vee(Sophus::SE3::hat(lie_se3)).transpose() << endl;

    //扰动
    cout << "特殊欧式群左扰动" << endl;

    Vector6d update_se3;

    update_se3.setZero();
    update_se3(0,0) = 1e-4d;
    Sophus::SE3 se3_updated = Sophus::SE3::exp(update_se3)*se3;
    cout << "添加左扰动之后se3是：\n" << se3_updated.matrix() << endl;

    return 0;
}