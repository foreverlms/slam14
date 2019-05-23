## rviz查看Octomap

由于ros indigo自带octomap 1.6,安装1.8会冲突,所以就直接用ros rviz来看octomap.

```shell
source .../ros_octomap/devel/setup.bash
roscore
```

```shell
catkin_make
source devel/setup.bash
rosrun octo_map_vis octomap_pub
```
```shell
rosrun rviz rviz
```

进入rviz后添加`OccupancyGrid`模块,然后选取topic为`octomap_msgs/map`即可.

**确保pose.txt和图像都在octomap_pub的运行目录下**