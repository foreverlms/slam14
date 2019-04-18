#ifndef MY_SLAM_FRAME_H
#define MY_SLAM_FRAME_H

#include <myslam/common_include.h>
#include <myslam/camera.h>
#include <opencv2/features2d/features2d.hpp>

namespace myslam
{
/**
 * 一帧图像所应包含的信息
 **/
class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;
    unsigned long id;
    double time_stamp;
    SE3 Tcw;
    //相机
    Camera::Ptr camera;
    Mat color, depth;

public:
    Frame() {}
    Frame(long id_, double time_stamp_ = 0, SE3 Tcw_ = SE3(), Camera::Ptr camera_ = nullptr, Mat color_ = Mat(), Mat depth_ = Mat());
    ~Frame();

    static Frame::Ptr createFrame();
    double findDepth(const cv::KeyPoint &kp);
    Vector3d getCameraCenter() const;
    bool isInFrame(const Vector3d &p_w);
};
} // namespace myslam

#endif // !MYSLAM_FRAME_H

