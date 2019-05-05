cmake_minimum_required(VERSION 2.8)
project(loop_closure)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OPencv: ${OpenCV_DIR}")
find_package(DBoW3 REQUIRED)
# set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
include_directories(${DBoW3_INCLUDE_DIRS})
# set( DBoW3_LIBS "/usr/local/lib/libDBoW3.a" )

add_executable(loop_closure loop_closure.cpp)
target_link_libraries(loop_closure  ${OpenCV_LIBS} ${DBoW3_LIBS})