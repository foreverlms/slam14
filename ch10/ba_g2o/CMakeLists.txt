cmake_minimum_required(VERSION 2.8)
project(ba_g2o)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(COMMON_PATH ${PROJECT_SOURCE_DIR}/../common)
add_subdirectory(${COMMON_PATH} common_bin)

Find_Package(G2O REQUIRED)
# Find_Package(Eigen3 REQUIRED)
Find_Package(Cholmod REQUIRED)

set(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator
             g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d 
             g2o_types_sba g2o_types_slam3d g2o_core g2o_interface 
             g2o_solver_csparse g2o_solver_structure_only 
             g2o_csparse_extension g2o_opengl_helper g2o_solver_dense 
             g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg 
             g2o_types_data g2o_types_sim3 cxsparse)


include_directories(${CHOLMOD_INCLUDE_DIR})
include_directories(include/ ceres/)

message("g2o头文件路径:" ${G2O_INCLUDE_DIR})
message("g2o库文件路径:" ${G2O_LIBS})
message("cholmod路径:" ${CHOLMOD_INCLUDE_DIR})


add_executable(g2o_bundle src/g2o_bundle.cpp)
target_link_libraries(g2o_bundle ${G2O_LIBS} ${CHOLMOD} ${CHOLMOD_LIBRARIES} BALProblem ParseCmd)