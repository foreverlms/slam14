#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <DBoW3/DBoW3.h>

void generating_dictionary(){
    std::cout << "读取图片..." << std::endl;
    std::vector<cv::Mat> images;
    for (size_t i = 0; i < 10; i++)
    {
        std::string path = "./data/" + std::to_string(i + 1) + ".png";
        cv::Mat image = cv::imread(path);

        images.push_back(image);
    }

    std::cout << "探测ORB特征点..." << std::endl;

    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    std::vector<cv::Mat> descriptors;
    for (size_t i = 0; i < 10; i++)
    {
        std::vector<cv::KeyPoint> kps;
        cv::Mat image;
        image = images[i];
        cv::Mat descriptor;
        // orb->detect(image, kps);
        // orb->compute(image, kps, descriptor);
        orb->detectAndCompute(image, cv::Mat(), kps, descriptor);
        descriptors.push_back(descriptor);
    }

    std::cout << "开始训练词袋..." << std::endl;
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    std::cout << "词袋:" << vocab << std::endl;

    vocab.save("vocabulary.yml.gz");
    std::cout << "词袋模型生成完毕!" << std::endl;
}