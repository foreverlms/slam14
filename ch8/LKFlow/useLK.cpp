#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "光流法需要您输入数据集路径" << endl;
        return 1;
    }

    string path = argv[1];
    string associate_file = path + "/associate.txt";
    ifstream fin(associate_file);
    string rgb_file, depth_file, time_rgb, time_depth;

    list<cv::Point2f> key_points; //提取的特征点

    cv::Mat color, depth, last_color;

    for (size_t index = 0; index < 9; index++)
    {
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        color = cv::imread(path + "/" + rgb_file);
        depth = cv::imread(path + "/" + depth_file);

        if (index == 0)
        {
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect(color, kps);
            for (auto kp : kps)
            {
                key_points.push_back(kp.pt);
            }
            last_color = color;
            //看看特征点的个数
            cout << key_points.size() << endl;
            continue;
        }

        if (color.data == NULL || depth.data == NULL)
        {
            continue;
        }

        vector<cv::Point2f> next_key_points, prev_key_points;
        for (auto kp : key_points)
            prev_key_points.push_back(kp);

        vector<unsigned char> status;
        vector<float> err;

        cv::calcOpticalFlowPyrLK(last_color, color, prev_key_points, next_key_points, status, err);

        int i = 0;
        for (auto iter = key_points.begin(); iter != key_points.end(); i++)
        {
            //status为0表示对应的像素点跟丢了，要删除掉
            if (status[i] == 0)
            {
                iter = key_points.erase(iter);
                continue;
            }

            *iter = next_key_points[i];
            iter++;
        }

        if (key_points.size() == 0){
            cout << "所有的关键点在光流法中都跟丢了" << endl;
            break;
        }

        //将追踪到的对应关键点标出来
        cv::Mat img_show = color.clone();
        for(auto kp : key_points)
            cv::circle(img_show,kp,10,cv::Scalar(0,240,0),1);

        cv::imshow("corners",img_show);
        cv::waitKey(0);
        last_color = color;
    }
    //看看删除之后特征点的个数
    cout << key_points.size() << endl;

    return 0;
}