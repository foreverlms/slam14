//
// Created by bob on 19-3-11.
//

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

void ch3_5(Eigen::MatrixXd &xd){
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i == j)
                xd(i,j) = 1;
            else
                xd(i,j) = 0;
        }
    }
}

int main(int argc,char * argv[]){

//    //第五题
//    Eigen::MatrixXd test_matrix = Eigen::MatrixXd::Random(5,5);
//
//    cout << "原矩阵：\n" << test_matrix << endl;
//
//    ch3_5(test_matrix);
//
//    cout << "提取左上角3*3块并赋值单位矩阵后的矩阵：\n" << test_matrix << endl;



    //第七题

    //小萝卜一号的位姿相对于世界坐标系变化
    Eigen::Quaterniond q1 = Eigen::Quaterniond(0.35,0.2,0.3,0.1);
    Eigen::Vector3d t1 = Eigen::Vector3d(0.3,0.1,0.1);

    //小萝卜一号的位姿相对于世界坐标系变化
    Eigen::Quaterniond q2 = Eigen::Quaterniond(-0.5,0.4,-0.1,0.2);
    Eigen::Vector3d t2 = Eigen::Vector3d(-0.1,0.5,0.3);

    //点p在世界坐标系下面的坐标是不变的，由此得到等式：Tcw1.inverse()*p1c = Tcw2.inverse()*p2c，p1c已知，求p2c

    Eigen::Vector3d p1c = Eigen::Vector3d(0.5,0,0.2);

    Eigen::Matrix3d r1 = q1.matrix();
    Eigen::Matrix3d r2 = q2.matrix();
    cout << "r1:\n" << r1 << endl;

    Eigen::Isometry3d tcw1 = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d tcw2 = Eigen::Isometry3d::Identity();

    tcw1.rotate(r1);
    tcw1.pretranslate(t1);
    cout << "变换矩阵1：\n" << tcw1.matrix() << endl;

    tcw2.rotate(r2);
    tcw2.pretranslate(t2);
    cout << "变换矩阵2：\n" << tcw2.matrix() << endl;

    Eigen::Vector3d p2c;
    p2c = tcw2*tcw1.inverse()*p1c;

    cout << "四元数未进行归一化：p点在小萝卜头二号坐标系下的坐标为：\n" << p2c.transpose() << endl;

    //四元数归一化
    q1.normalize();
    q2.normalize();

    r1 = q1.toRotationMatrix();
    r2 = q2.toRotationMatrix();
    cout << "r1:\n" << r1 << endl;

    tcw1.rotate(r1);
    tcw1.pretranslate(t1);
    cout << "变换矩阵1：\n" << tcw1.matrix() << endl;

    tcw2.rotate(r2);
    tcw2.pretranslate(t2);
    cout << "变换矩阵2：\n" << tcw2.matrix() << endl;

    p2c = tcw2*tcw1.inverse()*p1c;

    cout << "四元数进行归一化后：p点在小萝卜头二号坐标系下的坐标为：\n" << p2c.transpose() << endl;

    tcw1.rotate(q1);
    tcw1.pretranslate(t1);
    cout << "变换矩阵1：\n" << tcw1.matrix() << endl;

    tcw2.rotate(q2);
    tcw2.pretranslate(t2);
    cout << "变换矩阵2：\n" << tcw2.matrix() << endl;


    p2c = tcw2*tcw1.inverse()*p1c;

    cout << "四元数进行归一化后：p点在小萝卜头二号坐标系下的坐标为：\n" << p2c.transpose() << endl;
}