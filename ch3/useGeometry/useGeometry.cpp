//
// Created by bob on 19-3-10.
//

#include "useGeometry.h"


#include <iostream>
#include <cmath>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>



int main(int argc,char* argv[]){
    //旋转矩阵，先建立单位矩阵
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    //旋转向量[0,0,1],旋转角45度，也就是pi/4
    Eigen::AngleAxisd rotation_vector(M_PI/4,Eigen::Vector3d(0,0,1));

    cout.precision(3);
    //旋转向量转换为旋转矩阵 Rn = n，知道n可以求R,这个和下面的rotation_vector.matrix()一样
    rotation_matrix = rotation_vector.toRotationMatrix();

    cout << "rotation matrix = \n" << rotation_vector.matrix() << endl;


    //这个点要被旋转
    Eigen::Vector3d v(1,0,0);

    //通过旋转向量旋转
    Eigen::Vector3d v_rotated = rotation_vector * v;

    cout << "通过旋转向量旋转后的v: \n" << v_rotated.transpose() << endl;
    //通过欧拉角旋转
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0);

    cout << "偏航角yaw,俯仰角pitch,滚转角roll:" << euler_angles.transpose() << endl;

    //欧氏变换
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    //设置T的R
    T.rotate(rotation_vector);
    //设置T的t，平移
    T.pretranslate(Eigen::Vector3d(1,3,4));

    cout << "欧氏变换矩阵T是：\n" << T.matrix() << endl;

    Eigen::Vector3d v_transformed = T*v;

    cout << "v经过欧氏Tv变换后：\n" << v_transformed.transpose() << endl;

    //相似变换，这里用仿射变换代替
    Eigen::Affine3d sT = Eigen::Affine3d::Identity();
    
    sT.rotate(rotation_vector);

    sT.pretranslate(Eigen::Vector3d(1,3,4));
    //设定缩放比例2？
    sT.scale(2);


    Eigen::Vector3d v_scaled = sT*v;
    cout << "v经过相似变换后：\n" << v_scaled.transpose() << endl;

    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);

    cout << "quaternion四元数：\n" << q.coeffs() << endl;

    q = Eigen::Quaterniond(rotation_matrix);

    v_rotated = q * v;//这里数学上还是 v_rotated = qvq-1，这里乘法重载了

    cout << "通过四元数旋转后的v: \n" << v_rotated.transpose() << endl;


    return 0;
}