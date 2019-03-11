cmake_minimum_required(VERSION 3.13)
project(Slam14)

set(CMAKE_CXX_STANDARD 14)

include_directories(ch3/useEigen)

add_executable(Slam14 main.cpp ch3/useEigen/eigenMatrix.cpp ch3/useEigen/eigenMatrix.h)

add_executable(useGeometry ch3/useGeometry/useGeometry.cpp ch3/useGeometry/useGeometry.h)

add_executable(ch3_exercise ch3/ch3_exercise.cpp)