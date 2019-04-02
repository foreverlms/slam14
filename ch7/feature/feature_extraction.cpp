//
// Created by bob on 19-3-25.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

int main(int argc,char** argv){
    if ( argc != 3){
        cout << "请确保传入两张图像来做特征提取与匹配demo:feature_extraction img1_path img2_path" << endl;
        return 1;
    }

    cv::Mat img_1 = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2],CV_LOAD_IMAGE_COLOR);

    vector<cv::KeyPoint> kp_1,kp_2;

    cv::Mat descriptors_1,descriptors_2;
    //TODO 这里先这样改吧

//    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("ORB");
//    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::DescriptorExtractor::create("ORB");

    cv::Ptr<cv::ORB> orb = cv::ORB::create(500,1.2f,8,31,0,2,cv::ORB::HARRIS_SCORE,31,20);
    //通过ORB算法检测Oriented FAST角点
    orb->detect(img_1,kp_1);
    orb->detect(img_2,kp_2);

    //计算出相应角点的描述子
    orb->compute(img_1,kp_1,descriptors_1);
    orb->compute(img_2,kp_2,descriptors_2);

//    detector->detect(img_1,kp_1);
//    detector->detect(img_2,kp_2);
//
//    descriptor->compute(img_1,kp_1,descriptors_1);
//    descriptor->compute(img_2,kp_2,descriptors_2);

    cv::Mat outimag1;
    cv::namedWindow("ORB",cv::WINDOW_NORMAL);
    cv::drawKeypoints(img_1,kp_1,outimag1,cv::Scalar::all(-1));
    cv::imshow("ORB",outimag1);

    vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors_1,descriptors_2,matches);

    cout << "共找出" << matches.size() << "组匹配点" << endl;

    double min_dist = 1000, max_dist = 0;

    for (int i = 0; i < descriptors_1.rows; ++i) {
        double  dist = matches[i].distance;
        min_dist = min_dist > dist ? dist : min_dist;
        max_dist = max_dist < dist ? dist : max_dist;
    }

    printf("---最大汉明距离为：%f \n",max_dist);
    printf("---最小汉明距离为：%f \n",min_dist);

    vector<cv::DMatch> good_matches;

    for (int j = 0; j < descriptors_1.rows; ++j) {
        if (matches[j].distance <= max(2 * min_dist,30.0))
            good_matches.push_back(matches[j]);
    }

    cv::Mat img_matched;
    cv::Mat img_well_matched;

    cv::drawMatches(img_1,kp_1,img_2,kp_2,matches,img_matched);
    cv::drawMatches(img_1,kp_1,img_2,kp_2,good_matches,img_well_matched);


    cv::namedWindow("所有匹配点",cv::WINDOW_NORMAL);
    cv::namedWindow("筛选的匹配点",cv::WINDOW_NORMAL);
    cv::imshow("所有匹配点",img_matched);
    cv::imshow("筛选的匹配点",img_well_matched);
    cv::waitKey(0);

    return 0;
}