//
// Created by bob on 19/04/08.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;

/**
 * IMPORTANT!!!!!!!!!!!!
 * 这个书里的练习估计的是连续十张图片中的第二到第九张相对于第一张图片的位姿变换，而不是连续的前后帧间的比较！
 * 
 */
/**
 * 测量值，包含坐标与灰度
 */
struct Measurement
{
    Measurement(Eigen::Vector3d p, float g) : pos_world(p), grayscale(g) {}
    Eigen::Vector3d pos_world;
    float grayscale;
};
/**
 * 投影点像素坐标转相机坐标系下的坐标以及相机坐标系坐标转像素坐标 
 */
inline Eigen::Vector3d project2Dto3D(int x, int y, int d, float fx, float fy, float cx, float cy, float scale)
{
    float zz = float(d) / scale;
    float xx = zz * (x - cx) / fx;
    float yy = zz * (y - cy) / fy;
    return Eigen::Vector3d(xx, yy, zz);
}

inline Eigen::Vector2d project3Dto2D(float x, float y, float z, float fx, float fy, float cx, float cy)
{
    float u = fx * x / z + cx;
    float v = fy * y / z + cy;
    return Eigen::Vector2d(u, v);
}

bool poseEstimationDirect(const vector<Measurement> &measurements, cv::Mat *gray, Eigen::Matrix3f &intrinsics, Eigen::Isometry3d &Tcw);

class EdgeSE3ProjectDirect : public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect() {}

    //构造函数
    EdgeSE3ProjectDirect(Eigen::Vector3d point, float fx, float fy, float cx, float cy, cv::Mat *image)
        : x_world_(point), fx_(fx), fy_(fy), cx_(cx), cy_(cy), image_(image) {}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap *v = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        //通过位姿的估计值算出运动后p对应的世界坐标x_local
        Eigen::Vector3d x_local = v->estimate().map(x_world_);
        //x_local在相机坐标系中的归一化坐标
        float x = x_local[0] * fx_ / x_local[2] + cx_;
        float y = x_local[1] * fy_ / x_local[2] + cy_;

        //这里确保x_local对应的像素的位置必须在图像的4*4的一个窗口里面，感觉类似于LK光流法
        if (x - 4 < 0 || (x + 4) > image_->cols || (y - 4) < 0 || (y + 4) > image_->rows)
        {
            //如果该坐标不在图像的4*4块内，将误差置0
            _error(0, 0) = 0.0;
            this->setLevel(1);
        }
        else
        {
            _error(0, 0) = getPixelValue(x, y) - _measurement;
        }
    }

    virtual void linearizeOplus()
    {
        if (level() == 1)
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }

        g2o::VertexSE3Expmap *vtx = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        //书中的扰动分量q
        //pc及第二帧图片的坐标pw及第一帧图片坐标系下点的坐标
        //pc = Tcw * pw
        Eigen::Vector3d xyz_trans = vtx->estimate().map(x_world_);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        //u = (1 / z2) * K * q
        double invz_2 = invz * invz;
        float u = x * fx_ * invz + cx_;
        float v = y * fy_ * invz + cy_;

        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

        jacobian_uv_ksai(0, 0) = -x * y * invz_2 * fx_;
        jacobian_uv_ksai(0, 1) = fx_ * (1 + (x * x * invz_2));
        jacobian_uv_ksai(0, 2) = -y * invz * fx_;
        jacobian_uv_ksai(0, 3) = invz * fx_;
        jacobian_uv_ksai(0, 4) = 0;
        jacobian_uv_ksai(0, 5) = -x * invz_2 * fx_;

        jacobian_uv_ksai(1, 0) = -fy_ - fy_ * y * y * invz_2;
        jacobian_uv_ksai(1, 1) = x * y * fy_ * invz_2;
        jacobian_uv_ksai(1, 2) = fy_ * x * invz;
        jacobian_uv_ksai(1, 3) = 0;
        jacobian_uv_ksai(1, 4) = fy_ * invz;
        jacobian_uv_ksai(1, 5) = -fy_ * y * invz_2;

        Eigen::Matrix<double, 1, 2> jacobian_uv_pixel;

        //像素梯度
        jacobian_uv_pixel(0, 0) = (getPixelValue(u + 1, v) - getPixelValue(u - 1, v)) / 2; //水平方向
        jacobian_uv_pixel(0, 1) - (getPixelValue(u, v + 1) - getPixelValue(u, v - 1)) / 2; //竖直方向

        // J = - 像素梯度
        _jacobianOplusXi = jacobian_uv_pixel * jacobian_uv_ksai;
    }

    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}

  public:
    //x_world_是p在世界坐标系下的位置，在这里就是第一帧图片里的相机坐标系
    Eigen::Vector3d x_world_;
    float cx_ = 0, cy_ = 0, fx_ = 0, fy_ = 0;
    cv::Mat *image_ = nullptr;

  protected:
    inline float getPixelValue(float x, float y)
    {
        uchar *data = &image_->data[int(y) * image_->step + int(x)];
        float xx = x - floor(x);
        float yy = y - float(y);
        return float(
            //双线性插值计算灰度值，为何要用插值呢？
            (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[image_->step] + xx * yy * data[image_->step + 1]);
    }
};

bool poseEstimationDirect(const vector<Measurement> &measurements, cv::Mat *gray, Eigen::Matrix3f &intrinsics, Eigen::Isometry3d &Tcw){
    //求解优化的是T的李代数，因此是一个6维列向量
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    DirectBlock* solver_ptr = new DirectBlock(unique_ptr<DirectBlock::LinearSolverType>(linearSolver));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<DirectBlock>(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    //设定初始预估值
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(),Tcw.translation()));
    pose->setId(0);

    optimizer.addVertex(pose);

    int id =1;
    for (Measurement m : measurements){
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect(m.pos_world,intrinsics(0,0),intrinsics(1,1),intrinsics(0,2),intrinsics(1,2),gray);
        edge->setVertex(0,pose);
        //设定灰度测量值，用于与第index张图像灰度值相减来求误差
        edge->setMeasurement(m.grayscale);
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity());
        edge->setId(id++);
        optimizer.addEdge(edge);
    }

    cout << "直接法优化中的边数：" << optimizer.edges().size() << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    Tcw = pose->estimate();
}
int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "请为程序提供RGBD数据集路径" << '\n';
        return 1;
    }

    //让随机数更逼真
    srand((unsigned int) time(0));

    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate.txt";

    ifstream f_in(associate_file);

    string rgb_file, depth_file, time_rgb, time_depth;
    cv::Mat color, depth, gray;
    vector<Measurement> measurements;

    // 相机内参
    float cx = 325.5;
    float cy = 253.5;
    float fx = 518.0;
    float fy = 519.0;
    float depth_scale = 1000.0;
    Eigen::Matrix3f K;
    K << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.0f;

    //世界->相机的坐标系变换矩阵，这里以第一帧图像的相机坐标系为世界坐标系,接下来的一帧图片为相机坐标系
    //就是第一张图片->第index帧图片的坐标变换Tcw
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();//默认为单位阵

    cv::Mat prev_color;

    //每次只估第一张图片到第index张的运动
    for (int index = 0; index < 10; index++)
    {
        cout << "*********** loop " << index << " ************" << endl;
        f_in >> time_rgb >> rgb_file >> time_depth >> depth_file;
        color = cv::imread(path_to_dataset + "/" + rgb_file);
        depth = cv::imread(path_to_dataset + "/" + depth_file, -1);
        if (color.data == nullptr || depth.data == nullptr)
            continue;
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
        if (index == 0)
        {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> keypoints;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect(color, keypoints);
            for (auto kp : keypoints)
            {
                // 去掉邻近边缘处的点
                if (kp.pt.x < 20 || kp.pt.y < 20 || (kp.pt.x + 20) > color.cols || (kp.pt.y + 20) > color.rows)
                    continue;
                ushort d = depth.ptr<ushort>(cvRound(kp.pt.y))[cvRound(kp.pt.x)];
                if (d == 0)
                    continue;
                Eigen::Vector3d p3d = project2Dto3D(kp.pt.x, kp.pt.y, d, fx, fy, cx, cy, depth_scale);
                float grayscale = float(gray.ptr<uchar>(cvRound(kp.pt.y))[cvRound(kp.pt.x)]);
                measurements.push_back(Measurement(p3d, grayscale));
            }
            //记住第一张原始图片
            prev_color = color.clone();
            continue;
        }
        // 使用直接法计算相机运动
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        poseEstimationDirect(measurements, &gray, K, Tcw);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "直接法估计耗时：" << time_used.count() << "秒。" << endl;
        cout << "（第一张图片到当前图片的坐标变换矩阵）Tcw=" << endl << Tcw.matrix() << endl;

        // 画出特征点做对比，都是与第一张图片里的特征点位置对比
        cv::Mat img_show(color.rows * 2, color.cols, CV_8UC3);
        prev_color.copyTo(img_show(cv::Rect(0, 0, color.cols, color.rows)));
        color.copyTo(img_show(cv::Rect(0, color.rows, color.cols, color.rows)));
        for (Measurement m : measurements)
        {
            //点太多，这里是为了不输出那么多的点
            if (rand() > RAND_MAX / 5)
                continue;
            Eigen::Vector3d p = m.pos_world;
            Eigen::Vector2d pixel_prev = project3Dto2D(p(0, 0), p(1, 0), p(2, 0), fx, fy, cx, cy);
            Eigen::Vector3d p2 = Tcw * m.pos_world;
            Eigen::Vector2d pixel_now = project3Dto2D(p2(0, 0), p2(1, 0), p2(2, 0), fx, fy, cx, cy);
            if (pixel_now(0, 0) < 0 || pixel_now(0, 0) >= color.cols || pixel_now(1, 0) < 0 || pixel_now(1, 0) >= color.rows)
                continue;

            float b = 255 * float(rand()) / RAND_MAX;
            float g = 255 * float(rand()) / RAND_MAX;
            float r = 255 * float(rand()) / RAND_MAX;
            cv::circle(img_show, cv::Point2d(pixel_prev(0, 0), pixel_prev(1, 0)), 8, cv::Scalar(b, g, r), 2);
            cv::circle(img_show, cv::Point2d(pixel_now(0, 0), pixel_now(1, 0) + color.rows), 8, cv::Scalar(b, g, r), 2);
            cv::line(img_show, cv::Point2d(pixel_prev(0, 0), pixel_prev(1, 0)), cv::Point2d(pixel_now(0, 0), pixel_now(1, 0) + color.rows), cv::Scalar(b, g, r), 1);
        }
        cv::imshow("结果", img_show);
        cv::waitKey(0);
    }
    return 0;
}