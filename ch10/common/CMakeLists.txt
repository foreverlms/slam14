cmake_minimum_required(VERSION 2.8)
project(ba_g2o)

message(STATUS "${PROJECT_SOURCE_DIR} haha")
include_directories(./)

add_library(BALProblem SHARED BALProblem.cpp)
add_library(ParseCmd SHARED flags/command_args.cpp)