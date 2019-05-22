cmake_minimum_required(VERSION 2.8)
project(map)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -ggdb")

find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(-std=c++11)

add_subdirectory(dense_monocular)

add_subdirectory(dense_RGBD)