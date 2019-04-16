#ifndef MY_SLAM_CONFIG_H
#define MY_SLAM_CONFIG_H

#include <myslam/common_include.h>

namespace myslam
{
//Singleton单例模式
class Config
{
private:
    static std::shared_ptr<Config> config;
    cv::FileStorage file;

    Config() {}

public:
    ~Config();

    static void setParameterFile(const std::string &filename);
    template <typename T>
    static T get(const std::string &key)
    {
        return T(Config::config->file[key]);
    }
};
} // namespace myslam

#endif // !MY_SLAM_CONFIG_H
