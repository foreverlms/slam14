if [ ! -d "./build/" ];then
mkdir ./build
else
echo "build文件夹已存在"
fi

cd build
cmake ..
make rgbd
make install