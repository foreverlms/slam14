#include <iostream>
#include <vector>
#include <fstream>
#include <boost/timer.hpp>

#include <sophus/se3.h>
using Sophus::SE3;
using namespace std;
#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

//参数
const int boarder = 20;
const int width = 640;
const int height = 480;

//相机内参
const double fx = 481.2f;
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 2;                                              // NCC 取的窗口半宽度
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC窗口面积
const double min_cov = 0.1;                                                 // 收敛判定：最小方差
const double max_cov = 10;                                                  // 发散判定：最大方差

auto logger = spdlog::basic_logger_mt("basic_logger", "./log");

inline Vector3d px2cam(const Vector2d px)
{
    return Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1);
}

inline Vector2d cam2px(const Vector3d p_cam)
{
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy);
}

inline bool inside(const Vector2d &pt)
{
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

//双线性灰度插值,计算空间变换后的灰度
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt)
{
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) /
           255.0;
}

bool readDatasetFiles(const string &path, vector<string> &images, vector<SE3> &poses)
{
    ifstream fin(path + "/test_data/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin)
    {
        return false;
    }

    while (!fin.eof())
    {
        string image;
        fin >> image;
        double data[7];
        for (int i = 0; i < 7; i++)
        {
            fin >> data[i];
        }

        images.push_back(path + "/test_data/images/" + image);
        poses.push_back(
            SE3(Quaterniond(data[6], data[3], data[4], data[5]),
                Vector3d(data[0], data[1], data[2])));

        if (!fin.good())
        {
            break;
        }
    }

    return true;
}

void plotDepth(const Mat &depth)
{
    imshow("depth", depth * 0.4);
    waitKey(1);
}

// 显示极线匹配
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr)
{
    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}
// 显示极线
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr, const Vector2d &px_max_curr)
{
    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), Scalar(0, 255, 0), 1);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

// 计算 NCC 评分
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr)
{

    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_cur;

    //这里不能用size_t去声明i
    for (int i = -ncc_window_size; i <= ncc_window_size; i++)
    {
        for (int j = -ncc_window_size; j <= ncc_window_size; j++)
        {
            double ref_i_j = double(ref.ptr<uchar>(int(j + pt_ref(1, 0)))[int(i + pt_ref(0, 0))]) / 255.0;
            mean_ref += ref_i_j;

            //TODO:为何上面不做双线性插值?
            double curr_i_j = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(i, j));
            mean_curr += curr_i_j;

            values_ref.push_back(ref_i_j);
            values_cur.push_back(curr_i_j);
        }
    }

    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    double element_product = 0;
    double square_product_1 = 0;
    double square_product_2 = 0;
    for (int i = 0; i < values_ref.size(); i++)
    {
        element_product += (values_cur[i] - mean_curr) * (values_ref[i] - mean_ref);
        square_product_1 += (values_cur[i] - mean_curr) * (values_cur[i] - mean_curr);
        square_product_2 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
    }

    double ncc = element_product / sqrt(square_product_1 * square_product_2 + 1e-10); //1e-10是为了防止分母为0

    return ncc;
}

// 极线搜索
bool epipolarSearch(
    const Mat &ref,
    const Mat &curr,
    const SE3 &T_C_R,
    const Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Vector2d &pt_curr)
{
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize(); //归一化,L-2范数归一化
    //这个depth_mu是深度分布的均值?为何要乘以f_ref?深度只影响z啊
    //参考帧P向量
    Vector3d P_ref = f_ref * depth_mu;

    Vector2d px_mean_curr = cam2px(T_C_R * P_ref);
    //类似置信区间,取左右3\theta来估计最大最小
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
    d_min = d_min < 0.1 ? 0.1 : d_min;
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));

    Vector2d epipolar_line = px_max_curr - px_min_curr; //极线向量
    Vector2d epipolar_direction = epipolar_line;
    epipolar_direction.normalize(); //方向向量
    double half_length = 0.5 * epipolar_line.norm();
    //控制极线搜索范围
    if (half_length > 100)
    {
        half_length = 100;
    }

    // showEpipolarLine(ref, curr, pt_ref, px_min_curr, px_max_curr);

    double best_ncc = -1.0;
    Vector2d best_px_curr;

    for (double i = -half_length; i <= half_length; i += 0.7)
    {
        Vector2d px_curr = px_mean_curr + i * epipolar_direction;
        if (!inside(px_curr))
        {
            continue;
        }
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc)
        {
            best_ncc = ncc;
            best_px_curr = px_curr; //最佳匹配点
        }
    }
    if (best_ncc < 0.85f)
    {
        return false;
    }
    pt_curr = best_px_curr;
    // logger->info("pt_curr:({0:03.6f},{1:03.6f})",pt_curr[0],pt_curr[1]);

    return true;
}

// 更新深度滤波器
bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3 &T_C_R,
    Mat &depth,
    Mat &depth_cov)
{
    SE3 T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam(pt_ref);
    //归一化坐标
    f_ref.normalize();
    Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();

    //方程: d1*x1 = d2 * R_R_C *x2 + t_R_C,这里是d1*f_ref = d2*f2+t
    //书中通过等式两侧分别左乘f_ref^T和x2^T得到了两个线性方程,凑出了Ax=b的形式

    Vector3d t = T_R_C.translation();
    Vector3d x2 = T_R_C.rotation_matrix() * f_curr;
    //b向量
    Vector2d b(t.dot(f_ref), t.dot(x2));
    //系数矩阵A{4}
    double A[4];
    A[0] = f_ref.dot(f_ref);
    A[1] = -f_ref.dot(x2);
    //A[1]=-A[2]
    A[2] = x2.dot(f_ref);
    A[3] = -x2.dot(x2);

    //根据克拉默法则x_i = D_i /D
    double D = A[0] * A[3] - A[1] * A[2];
    double D_1 = b[0] * A[3] - A[1] * b[1];
    double D_2 = A[0] * b[1] - b[0] * A[2];

    Vector2d depth_solution = Vector2d(D_1, D_2) / D;

    //这里p_ref_with_depth_1和p_ref_with_depth_2都是ref帧下的深度三角测量值
    Vector3d p_ref_with_depth_1 = depth_solution[0] * f_ref;
    Vector3d p_ref_with_depth_2 = depth_solution[1] * x2 + t;
    //去二者平均作为最终三角测量深度向量
    Vector3d p_ref_with_depth = (p_ref_with_depth_1 + p_ref_with_depth_2) / 2.0;
    double depth_estimation_ref = p_ref_with_depth.norm();

    //计算不确定性
    Vector3d p_ref = f_ref * depth_estimation_ref;
    Vector3d a = p_ref - t;

    //对照P327图13-4
    double t_norm = t.norm(), a_norm = a.norm();
    //alpha角的值,弧度制
    //f_ref已经归一化,其norm值为1,相当于alpha = p_ref.dot(t)/(t.norm()*p_ref.norm())
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(a.dot(t) / (t_norm * a_norm));
    double delta_belta = atan(1.0 / fx);
    double beta_prime = beta + delta_belta;

    double gamma = M_PI - alpha - beta_prime;
    double p_prime_norm = t_norm * sin(beta_prime) / sin(gamma);
    //TODO:书中这样算的
    //double d_cov = p_prime_norm - depth_estimation_ref;
    double d_cov = depth_estimation_ref - p_prime_norm;
    double d_cov_square = d_cov * d_cov;

    //高斯融合
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma_square = depth_cov.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

    double mu_fuse = (d_cov_square * mu + sigma_square * depth_estimation_ref) / (sigma_square + d_cov_square);
    double sigma_fuse_square = (sigma_square * d_cov_square) / (d_cov_square + sigma_square);

    //更新深度分布
    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov.ptr<double>(int(pt_ref(0, 0)))[int(pt_ref(1, 0))] = sigma_fuse_square;

    return true;
}

/**
 * 更新图像深度
 */
bool update(const cv::Mat &ref, const cv::Mat &curr, const SE3 &T_C_R, cv::Mat &depth, cv::Mat &depth_cov)
{
// openmp用法
#pragma omp parallel for
    for (int x = boarder; x < width - boarder; x++)
    {
        // logger->info("x: {}",x);
#pragma omp parallel for
        for (int y = boarder; y < height - boarder; y++)
        {
            // logger->info("y: {}", y);
            if (depth_cov.ptr<double>(y)[x] < min_cov || depth_cov.ptr<double>(y)[x] > max_cov)
            {
                continue;
            }

            Vector2d pt_curr;
            bool ret = epipolarSearch(
                ref, curr, T_C_R, Vector2d(x, y), depth.ptr<double>(y)[x], sqrt(depth_cov.ptr<double>(y)[x]), pt_curr);
            if (!ret)
            {
                continue;
            }
            showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);
            updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, depth, depth_cov);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "需要指定数据集路径!" << endl;
        return -1;
    }

    // spdlog::logger logger();
    
    // try
    // {
    //     auto logger = spdlog::basic_logger_mt("basic_logger","logs/log.txt");
    // }
    // catch(const spdlog::spdlog_ex& ex)
    // {
    //     std::cout << "日志初始化失败!" << ex.what() << std::endl;
    // }
    
    
    
    vector<string> color_image_files;
    vector<SE3> poses_T_W_C;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_T_W_C);
    if (!ret)
    {
        cout << "读取图片失败!" << endl;
        return -1;
    }
    cout << "一共获得" << color_image_files.size() << "张图片." << endl;
    Mat ref = imread(color_image_files[0], 0);
    SE3 pose_ref_T_W_C = poses_T_W_C[0];

    double init_depth = 3.0;
    double init_cov_square = 3.0;

    Mat depth(height, width, CV_64F, init_depth);
    Mat depth_cov(height, width, CV_64F, init_cov_square);

    for (int index = 1; index < color_image_files.size(); index++)
    {
        cout << "loop:" << index << endl;
        cout << color_image_files[index] << endl;
        Mat curr = imread(color_image_files[index], 0);
        if (curr.data == nullptr)
        {
            continue;
        }

        SE3 pose_curr_T_W_C = poses_T_W_C[index];
        SE3 pose_T_C_R = pose_curr_T_W_C.inverse() * pose_ref_T_W_C;
        update(ref, curr, pose_T_C_R, depth, depth_cov);
        plotDepth(depth);
        imshow("images", curr);
        waitKey(1);

        cout << "index: " << index << endl;
        if(index == 15){
            break;
        }
    }
    cout << "估计完成,保存结果中..." << endl;
    imwrite("depth.png", depth);
    cout << "程序结束" << endl;
    return 0;
}