#include <myslam/config.h>

namespace myslam
{
void Config::setParameterFile(const std::string &filename)
{
    if (config == nullptr)
    {
        config = shared_ptr<Config>(new Config());
    }
    std::cout << "文件路径:"<<filename << endl;
    config->file = cv::FileStorage(filename, cv::FileStorage::READ);

    if (!config->file.isOpened())
    {
        cerr << "参数文件：" << filename << "不存在!" << endl;
        config->file.release();
    }
    return;
}

Config::~Config()
{
    if (config->file.isOpened())
    {
        file.release();
    }
}

shared_ptr<Config> Config::config = nullptr;

} // namespace myslam
