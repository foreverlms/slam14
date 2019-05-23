cmake_minimum_required(VERSION 2.8.3)
project(octo_map_vis)

add_definitions(-std=c++11)

find_package(octomap 1.6 REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})
# find_package(octomap_msgs REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  octomap_msgs
)

find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


catkin_package(
 INCLUDE_DIRS include
 LIBRARIES octo_map_vis
 CATKIN_DEPENDS octomap_msgs roscpp rospy std_msgs
 DEPENDS system_lib octomap OpenCV
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


add_executable(octomap_pub src/octomap.cpp)
target_link_libraries(octomap_pub ${catkin_LIBRARIES} ${OCTOMAP_LIBRARIES} ${OpenCV_LIBRARIES})
