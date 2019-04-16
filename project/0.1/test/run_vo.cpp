#include <myslam/visual_odometry.h>
#include <myslam/config.h>
#include <fstream>
#include <opencv2/viz.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/timer.hpp>

using namespace myslam;

int main(int argc,char** argv){
    if(argc != 2){
        cout << "需要指定视觉里程计的参数文件！" <<endl;
        return -1;
    }

    Config::setParameterFile(argv[1]);
    VisualOdometry::Ptr vo(new VisualOdometry);

    string data_set = Config::get<string>("dataset_dir");
    cout << "数据集路径：" << data_set << endl;

    ifstream fin(data_set+"/associate.txt");

    if (!fin)
    {
        cout << "请确保associate.txt存在！" << endl;
    }
    vector<string> rgb_files,depth_files;
    vector<double> rgb_times,depth_times;
    //fstream类重写了　!　运算符
    while (!fin.eof()) 
    {
        string rgb_time,rgb_file,depth_time,depth_file;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file;
        rgb_times.push_back(atof(rgb_time.c_str()));
        depth_times.push_back(atof(depth_time.c_str()));

        rgb_files.push_back(data_set+"/"+rgb_file);
        depth_files.push_back(data_set+"/"+depth_file);

        if (fin.good() == false)
        {
            break;
        }
        
    }

    Camera::Ptr camera(new Camera);

    cv::viz::Viz3d vis("视觉里程计");
    cv::viz::WCoordinateSystem world_coor(1.0),camera_coor(0.5);
    //相机位置，光心位置，相机y方向？
    cv::Point3d cam_pos(0,-1.0,-1.0),cam_focal_point(0,0,0),cam_y_dir(0,1,0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos,cam_focal_point,cam_y_dir);

    vis.setViewerPose(cam_pose);
    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH,2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH,1.0);
    vis.showWidget("world",world_coor);
    vis.showWidget("camera",camera_coor);

    cout << "共有" << rgb_files.size() << "帧图片" <<endl;
    for(int i=0;i<rgb_files.size();i++){
        Mat color = cv::imread(rgb_files[i]);
        Mat depth = cv::imread(depth_files[i],-1);

        if(color.data == nullptr || depth.data == nullptr){
            break;
        }
        Frame::Ptr frame = Frame::createFrame();
        frame->camera = camera;
        frame->color = color;
        frame->depth = depth;
        frame->time_stamp = rgb_times[i];
        
        boost::timer timer;
        vo->addFrame(frame);
        cout << "视觉里程计耗时：" << timer.elapsed() << endl;

        if (vo->state == VisualOdometry::LOST){
            break;
        }

        //这里为什么要取逆？
        //因为这里Tcw是世界坐标系到相机坐标系的变换，现在要把相机坐标系显示在世界坐标系里，所以反过来取逆
        SE3 Tcw = frame->Tcw.inverse();
        cv::Affine3d M(
            cv::Affine3d::Mat3(
                Tcw.rotation_matrix()(0,0),Tcw.rotation_matrix()(0,1),Tcw.rotation_matrix()(0,2),
                Tcw.rotation_matrix()(1,0),Tcw.rotation_matrix()(1,1),Tcw.rotation_matrix()(1,2),
                Tcw.rotation_matrix()(2,0),Tcw.rotation_matrix()(2,1),Tcw.rotation_matrix()(2,2)
                ),
                cv::Affine3d::Vec3(
                    Tcw.translation()(0,0),
                    Tcw.translation()(1,0),
                    Tcw.translation()(2,0))
        );

        cv::imshow("image",color);
        cv::waitKey(1);
        vis.setWidgetPose("camera",M);
        vis.spinOnce(1, false);
    }
    return 0;
}