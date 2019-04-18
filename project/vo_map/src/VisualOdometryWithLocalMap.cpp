#include <myslam/visual_odometry_with_localmap.h>
#include <boost/timer.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <myslam/g2o_types.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

namespace myslam
{

VisualOdometryWithLocalMap::VisualOdometryWithLocalMap() : VisualOdometry(), matcher_flann(new cv::flann::LshIndexParams(5, 10, 2))
{
    map_point_erase_ratio = Config::get<double>("map_point_erasse_ratio");
}
VisualOdometryWithLocalMap::~VisualOdometryWithLocalMap() = default;
void VisualOdometryWithLocalMap::addKeyFrame()
{
    if (map->key_frames.empty())
    {
        for (size_t i = 0; i < kps_curr.size(); i++)
        {
            double d = curr->findDepth(kps_curr[i]);
            if (d < 0)
            {
                continue;
            }

            Vector3d p_w = ref->camera->pixel2world(Vector2d(kps_curr[i].pt.x, kps_curr[i].pt.y), curr->Tcw, d);
            //TODO: 这里n是干嘛的？
            Vector3d n = p_w - ref->getCameraCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(p_w, n, curr.get(), descriptors_curr.row(i));
            map->insertMapPoint(map_point);
        }
    }
    map->insertKeyFrame(curr);
    ref = curr;
}

bool VisualOdometryWithLocalMap::addFrame(Frame::Ptr ptr)
{
    switch (state)
    {
    case INITIALIZING:
        state = OK;
        curr = ref = ptr;

        //求解特征点
        extractKeyPoints();
        //计算描述子
        computeDescriptors();
        //第一张肯定是关键帧
        addKeyFrame();

        //使用地图的vo不再比较当前帧与参考帧（前一帧）
        //setRef3DPoints();
        break;

    case OK:
    {
        curr = ptr;
        curr->Tcw = ref->Tcw;
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
        if (checkEstimatePose())
        {
            curr->Tcw = Tcw_estimated;
            optimizeMap();
            // ref = curr;
            // setRef3DPoints();
            //确保只有连续丢失max_num_lost帧才会设定当前状态state为LOST
            num_lost = 0;
            if (checkKeyFrame())
            {
                addKeyFrame();
            }
        }
        else
        {
            cout << "连续已跟丢的帧数是：" << num_lost << endl;
            num_lost++;
            if (num_lost > max_num_lost)
            {
                state = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        std::cout << "视觉里程计已经丢失信息，无法继续进行位姿估计。" << std::endl;
        break;
    }
    }

    return true;
}

void VisualOdometryWithLocalMap::optimizeMap()
{

    for (auto iter = map->map_points.begin(); iter != map->map_points.end();)
    {
        //删除不在视野之内的点？
        if (!curr->isInFrame(iter->second->pos))
        {
            iter = map->map_points.erase(iter);
            continue;
        }

        //随着相机运动匹配率慢慢不是那么高的点
        float match_ratio_ = float(iter->second->matched_times) / iter->second->visible_times;
        if (match_ratio_ < map_point_erase_ratio)
        {
            iter = map->map_points.erase(iter);
            continue;
        }

        double angle = getViewAngle(curr, iter->second);
        if (angle > M_PI / 6.)
        {
            iter = map->map_points.erase(iter);
            continue;
        }

        if (!iter->second->good)
        {
            //TODO: 三角化？啥意思？
        }

        iter++;
    }

    if (match_2d_kp_index.size() < 100)
    {
        addMapPoints();
    }

    if (map->map_points.size() > 1000)
    {
        //TODO:去除多余点
        map_point_erase_ratio += 0.05;
    }
    else
    {
        map_point_erase_ratio = 0.1;
    }

    cout << "地图优化之后点数：" << map->map_points.size() << endl;
}
double VisualOdometryWithLocalMap::getViewAngle(Frame::Ptr frame, MapPoint::Ptr point)
{
    Vector3d n = point->pos - frame->getCameraCenter();
    n.normalize();
    //TODO: What the fuck?
    return acos(n.transpose() * point->norm);
}
void VisualOdometryWithLocalMap::poseEstimationPnp()
{
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;

    for (int index : match_2d_kp_index)
    {
        pts2d.push_back(kps_curr[index].pt);
    }
    for (auto pt : match_3dpts)
    {
        pts3d.push_back(pt->getPositionCV());
    }

    Mat K = (cv::Mat_<double>(3, 3) << ref->camera->fx, 0, ref->camera->cx,
             0, ref->camera->fy, ref->camera->cy,
             0, 0, 1);

    Mat rvec, tvec, inliers;
    cv::solvePnPRansac(pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
    num_inliers = inliers.rows;
    cout << "PnPRANSAC局类点个数：" << num_inliers << endl;
    Tcw_estimated = SE3(
        SO3(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0)),
        Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block;
    Block::LinearSolverType *linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block *solver_ptr = new Block(unique_ptr<Block::LinearSolverType>(linearSolver));
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<Block>(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(
        Tcw_estimated.rotation_matrix(), Tcw_estimated.translation()));
    optimizer.addVertex(pose);

    //边
    for (size_t i = 0; i < inliers.rows; i++)
    {
        int index = inliers.at<int>(i, 0);
        EdgeProjectXYZ2UVPoseOnly *edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId(i);
        edge->setVertex(0, pose);
        edge->camera = curr->camera.get();
        edge->point = Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z);
        edge->setMeasurement(Eigen::Vector2d(pts2d[index].x, pts2d[index].y));
        edge->setInformation(Eigen::Matrix2d::Identity());

        optimizer.addEdge(edge);
        //记录下对应３d点
        //这里是真没想明白？为啥pts3d的index和match_3dpts一样？
        match_3dpts[index]->matched_times++;
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    Tcr_estimated = SE3(pose->estimate().rotation(), pose->estimate().translation());
}

bool VisualOdometryWithLocalMap::checkEstimatedPose()
{
    if (num_inliers < min_inliers)
    {
        cout << "局类点数太少，本次位姿估计失效！" << endl;
        return false;
    }

    //这两种方式是一样的
    // SE3 Tr_c = ref->Tcw*Tcw_estimated.inverse();
    SE3 Tc_r = Tcw_estimated.inverse() * ref->Tcw.inverse();
    Sophus::Vector6d d = Tc_r.log();
    if (d.norm() > 5.0)
    {
        cout << "本次位姿估计太大，失效！" << endl;
        return false;
    }
    return true;
}
bool VisualOdometryWithLocalMap::checkKeyFrame()
{
    SE3 Tc_r = Tcw_estimated.inverse() * ref->Tcw.inverse();
    Sophus::Vector6d d = Tc_r.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    //是否满足关键帧的条件
    if (rot.norm() > key_frame_min_rot || trans.norm() > key_frame_min_trans)
        return true;
    return false;
}
void VisualOdometryWithLocalMap::featureMatching()
{
    //这里是不是取地图里面有而且出现在当前帧里的点？
    boost::timer timer;
    vector<cv::DMatch> matches;
    Mat desp_map;
    vector<MapPoint::Ptr> candidate;
    for (auto &allpoints : map->map_points)
    {
        MapPoint::Ptr &p = allpoints.second;
        if (curr->isInFrame(p->pos))
        {
            //p在当前帧里出现过，次数加一
            p->visible_times++;
            candidate.push_back(p);
            desp_map.push_back(p->descriptor);
        }
    }

    matcher_flann.match(desp_map, descriptors_curr, matches);
    float min_dis = std::min_element(matches.begin(), matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2) { return m1.distance < m2.distance; })->distance;

    match_3dpts.clear();
    match_2d_kp_index.clear();

    for (cv::DMatch &m : matches)
    {
        if (m.distance < std::max<float>(min_dis * match_ratio, 30.0))
        {
            //记下地图里匹配上的点
            match_3dpts.push_back(candidate[m.queryIdx]);
            //记下当前帧匹配的2D点
            match_2d_kp_index.push_back(m.trainIdx);
        }
    }
    cout << "优良匹配点个数：" << match_3dpts.size() << endl;
    cout << "匹配耗时：" << timer.elapsed() << endl;
}
/**
 * 将当前帧里的一些新点加入到地图中
 */
void VisualOdometryWithLocalMap::addMapPoints()
{
    vector<bool> matched(kps_curr.size(), false);
    for (auto index : match_2d_kp_index)
    {
        matched[index] = true;
    }
    for (int i = 0; i < kps_curr.size(); i++)
    {
        //说明这个点之前已经在地图里面记录了
        if (matched[i])
        {
            continue;
        }
        double d = ref->findDepth(kps_curr[i]);
        //深度为０的点，当然不记录
        if (d < 0)
        {
            continue;
        }

        Vector3d p_w = ref->camera->pixel2world(Vector2d(kps_curr[i].pt.x, kps_curr[i].pt.y), curr->Tcw, d);
        Vector3d n = p_w - ref->getCameraCenter();
        n.normalize();
        MapPoint::Ptr map_point = MapPoint::createMapPoint(p_w, n, curr.get(), descriptors_curr.row(i).clone());
        map->insertMapPoint(map_point);
    }
}
} // namespace myslam
