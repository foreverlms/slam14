#include <myslam/common_include.h>
#include <myslam/frame.h>
#include <myslam/map.h>
#include <myslam/visual_odometry.h>
#include <myslam/config.h>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <myslam/g2o_types.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

namespace myslam {
    VisualOdometry::VisualOdometry() : state(INITIALIZING), ref(nullptr), curr(nullptr), map(new Map()), num_lost(0),
                                       num_inliers(0) {
        num_of_features = Config::get<int>("number_of_features");
        scale_factor = Config::get<double>("scale_factor");
        level_pyramid = Config::get<int>("level_pyramid");
        match_ratio = Config::get<float>("match_ratio");
        key_frame_min_rot = Config::get<double>("keyframe_rotation");
        key_frame_min_trans = Config::get<double>("keyframe_translation");
        max_num_lost = Config::get<int>("max_num_lost");
        min_inliers = Config::get<int>("min_inliers");

        orb = cv::ORB::create(num_of_features, scale_factor, level_pyramid);
    }

    VisualOdometry::~VisualOdometry() = default;

    bool VisualOdometry::addFrame(Frame::Ptr ptr) {
        switch (state) {
            case INITIALIZING:
                state = OK;
                curr = ref = ptr;
                //第一张肯定是关键帧
                map->insertKeyFrame(ptr);
                extractKeyPoints();
                computeDescriptors();

                setRef3DPoints();
                break;

            case OK: {
                curr = ptr;
                extractKeyPoints();
                computeDescriptors();
                featureMatching();
                poseEstimationPnP();
                if (checkEstimatePose()) {
                    curr->Tcw = Tcr_estimated * ref->Tcw;
                    ref = curr;
                    setRef3DPoints();
                    //确保只有连续丢失max_num_lost帧才会设定当前状态state为LOST
                    num_lost = 0;
                    if (checkKeyFrame()) {
                        addKeyFrame();
                    }
                } else {
                    cout << "连续已跟丢的帧数是：" << num_lost << endl;
                    num_lost++;
                    if (num_lost > max_num_lost) {
                        state = LOST;
                    }
                    return false;
                }
                break;
            }
            case LOST: {
                std::cout << "视觉里程计已经丢失信息，无法继续进行位姿估计。" << std::endl;
                break;
            }
        }

        return true;
    }

/**
 * 将当前帧加入到关键帧中
 */
    void myslam::VisualOdometry::addKeyFrame() {
        cout << "像地图中添加一幅关键帧" << endl;
        map->insertKeyFrame(curr);
    }

    void VisualOdometry::extractKeyPoints() {
        orb->detect(curr->color, kps_curr);
    }

    void VisualOdometry::computeDescriptors() {
        orb->compute(curr->color, kps_curr, descriptors_curr);
    }

    void VisualOdometry::featureMatching() {
        vector<cv::DMatch> matches;
        cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING);
        matcher.match(descriptors_ref, descriptors_curr, matches);

        //筛选匹配点
        float min_dist = std::min_element(matches.begin(), matches.end(),
                                          [](const cv::DMatch &m1, const cv::DMatch &m2) -> bool {
                                              return m1.distance < m2.distance;
                                          })->distance;

        //清空上两帧的匹配
        feature_matches.clear();
        for (auto m : matches) {
            //筛选匹配
            if (m.distance < max<float>(min_dist * match_ratio, 30.0)) {
                feature_matches.push_back(m);
            }
        }
        cout << "一共得到" << feature_matches.size() << "对匹配点" << endl;
    }

    void VisualOdometry::poseEstimationPnP() {
        vector<cv::Point3f> pts3D;
        vector<cv::Point2f> pts2D;

        for (auto m : feature_matches) {
            //这里正确
            pts3D.push_back(pts_3d_ref[m.queryIdx]);
            pts2D.push_back(kps_curr[m.trainIdx].pt);
        }

        //相机矩阵K
        Mat K = (cv::Mat_<double>(3, 3) << ref->camera->fx, 0, ref->camera->cx,
                0, ref->camera->fy, ref->camera->cy,
                0, 0, 1);
        Mat rvec, tvec, inliers;//旋转向量，平移向量

        //solvePnpRansac在PnP求解之前进行了Ransac处理，更准确．
        cv::solvePnPRansac(pts3D, pts2D, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
        num_inliers = inliers.rows;
        cout << "pnp方法的局内点是：" << num_inliers << "个" << endl;

        Tcr_estimated = SE3(SO3(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0)),
                            Eigen::Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));

        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
        Block::LinearSolverType* linear_slover_type = new g2o::LinearSolverDense<Block::PoseMatrixType>();
        Block* solve_ptr = new Block(unique_ptr<Block::LinearSolverType>(linear_slover_type));
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<Block>(solve_ptr));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
        pose->setId(0);
        pose->setEstimate(g2o::SE3Quat(Tcr_estimated.rotation_matrix(),Tcr_estimated.translation()));
        optimizer.addVertex(pose);

        for (size_t i = 0; i < inliers.rows; i++)
        {
            int index = inliers.at<int>(i,0);
            EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
            edge->setId(i);
            edge->setVertex(0,pose);
            edge->camera = curr->camera.get();
            edge->point = Vector3d(pts3D[index].x,pts3D[index].y,pts3D[index].z);
            edge->setMeasurement(Eigen::Vector2d(pts2D[index].x, pts2D[index].y));
            edge->setInformation(Eigen::Matrix2d::Identity());

            optimizer.addEdge(edge);
        }
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        Tcr_estimated = SE3(pose->estimate().rotation(), pose->estimate().translation());
    }

    void VisualOdometry::setRef3DPoints() {
        //筛选点并且只考虑有效点的描述子
        pts_3d_ref.clear();
        descriptors_ref = Mat();
        for (size_t i = 0; i < kps_curr.size(); i++) {
            double d = ref->findDepth(kps_curr[i]);
            if (d > 0) {
                //获取参考帧3D点
                Vector3d p_3d = ref->camera->pixel2camera(Vector2d(kps_curr[i].pt.x, kps_curr[i].pt.y), d);
                pts_3d_ref.push_back(cv::Point3f(p_3d(0), p_3d(1), p_3d(2)));
                //更新参考帧描述子
                descriptors_ref.push_back(descriptors_curr.row(i));
            }

        }
    }

    bool VisualOdometry::checkEstimatePose() {
        //先判断位姿估计是否满足了预定要求的点数
        if (num_inliers < min_inliers) {
            cout << "当前的位姿估计因为局内点太少不符合模型而失效！" << endl;
            return false;
        }
        //再判断估计出来的位姿是不是超过了两帧之间最大的运动量
        Sophus::Vector6d d = Tcr_estimated.log();
        if (d.norm() > 5.0) {
            cout << "当前的位姿估计因为运动太大而失效！" << endl;
            return false;
        }

        return true;

    }

    bool VisualOdometry::checkKeyFrame() {
        Sophus::Vector6d d = Tcr_estimated.log();
        //sophus中平移向量在前，旋转向量在后
        Vector3d trans = d.head<3>();
        Vector3d rot = d.tail<3>();

        return rot.norm() > key_frame_min_rot || trans.norm() > key_frame_min_trans;
    }

} // namespace myslam
