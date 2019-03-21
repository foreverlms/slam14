//
// Created by bob on 19-3-20.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>
#include <vector>

using namespace std;

struct CURVE_FITTING_COST{
    const double _x,_y;
    CURVE_FITTING_COST (double x, double y) : _x(x),_y(y){};

    //残差
    template <typename T>
    bool operator()( const T* const abc, T* residual) const {
        residual[0] = T(_y) - ceres::exp(abc[0]*T(_x) *T(_x)+abc[1]*T(_x)+abc[2]);
        return true;
    }
};
int main(int argc,char** argv){
    double a=1.0,b=2.0,c=1.0;
    int N = 100;
    //噪声的标准差
    double w_sigma = 1.0;
    cv::RNG rng;
    //对a,b.c的估计值
    double abc[3] = {0,0,0};

    vector<double> x_data,y_data;//实际的x,y数据值

    cout << "正在随机生成x,y的数据:" << endl;

    for (int i = 0; i < N; ++i) {
        double  x = i /100.0;
        x_data.push_back(x);
        y_data.push_back(exp(a * x * x+b*x+c)+rng.gaussian(w_sigma));

        cout << x_data[i] << " " << y_data[i] << endl;
    }

    ceres::Problem problem;
    for (int j = 0; j < N; ++j) {
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,3>(new CURVE_FITTING_COST(x_data[j],y_data[j])),
                nullptr,abc
                );
    }

    ceres::Solver::Options options;
    //设置线性增量方程的解法
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    ceres::Solve(options,&problem,&summary);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "优化求解过程耗费时间： " << time_used.count() << "秒。" << endl;

    cout << summary.BriefReport() << endl;

    cout << "估计的相关参数a,b,c为：";
    for(auto tmp : abc)
        cout << tmp << " ";

    cout << endl;

    return 0;

}
