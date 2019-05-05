#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "DBoW3/DBoW3.h"

using namespace std;

// void generating_dictionary();

int main(int argc,char** argv){
    cout << "读取词袋..." << endl;
    DBoW3::Vocabulary vocab("./vocabulary.yml.gz");
    if (vocab.empty())
    {
        cerr << "词袋读入存在问题!" << endl;
        return 1;
    }

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

    cout << "图片与图片间比较..." << endl;
    for (size_t i = 0; i < images.size(); i++)
    {
        //单词向量v1
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i],v1);
        for (size_t j = 0; j < images.size(); j++)
        {
            //单词向量v2
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j],v2);
            //计算score
            double score = vocab.score(v1,v2);
            cout << "图像" << i<< "和图像" << j <<"相似度得分:" << score << endl;
        }

        cout<<endl;
        
    }
    
    cout << "通过数据库比较" << endl;
    DBoW3::Database db(vocab,false,0);
    for (size_t i = 0; i < descriptors.size(); i++)
    {
        db.add(descriptors[i]);
    }

    cout << "数据库信息" << db << endl;
    for (size_t j = 0; j < descriptors.size(); j++)
    {
        DBoW3::QueryResults result;
        db.query(descriptors[j],result,4);
        cout << "找寻图片" << j << "返回" << result << endl;
    }
}