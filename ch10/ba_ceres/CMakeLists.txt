cmake_minimum_required(VERSION 2.8)
project(ba_ceres)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(COMMON_PATH ${PROJECT_SOURCE_DIR}/../common)
add_subdirectory(${COMMON_PATH} common_bin)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS}
                    ${COMMON_PATH}
                    ${COMMON_PATH}/tools               
                    ${COMMON_PATH}/flags)               


add_executable(ceres_ba ceres_bundle.cpp)
target_link_libraries(ceres_ba BALProblem ParseCmd ${CERES_LIBRARIES})