//
// Created by bob on 19-3-16.
//

#include <iostream>
#include <chrono>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc,char* argv[]){
    cv::Mat image;

    image = cv::imread(argv[1]);

    if (image.data != nullptr){
        cerr << "您指定的文件路径：" << argv[1] << "不存在。" << endl;
        return 0;
    }

    cout << "图像宽度为：" << image.cols << "；高度为：" << image.rows << "。通道数为：" << image.channels() << endl;
    cv::imshow("image",image);
    cv::waitKey(0);

    if (image.type() != CV_8UC1 && image.type() != CV_8UC3){
        cout << "请确保您使用的图片是灰度图或者彩色图" << endl;
        return 0;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    for (size_t y = 0; y < image.rows ; y++) {
        for (size_t x = 0; x < image.cols; x++){
            //unsigned char* row_ptr = image.ptr<unsigned cahr>(y);
            //row_ptr 是第y行的头指针
            unsigned char* row_ptr = image.ptr(y);
            unsigned char* data_ptr = &row_ptr[x * image.channels()];
            for (int i = 0; i < image.channels(); ++i) {
                unsigned char data = data_ptr[i];
            }
        }
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "遍历图像所有数据点用时" << time_used.count() << "秒" << endl;

    //直接赋值不会拷贝图像，引用赋值
    cv::Mat image_ref = image;

    image_ref(cv::Rect(0,0,100,100)).setTo(0);

    cv::imshow("image",image);
    cv::waitKey(0);

    //深拷贝
    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0,0,100,100)).setTo(255);
    cv::imshow("image_clone",image_clone);
    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}