cmake_minimum_required(VERSION 2.8)

set(OpenCV_DIR /opt/opencv344/share/OpenCV)

project(Slam14)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(CMAKE_CXX_STANDARD 14)
#指定Sophus package路径
set(Sophus_DIR ch4/useSophus/Sophus/build )

#list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/ch6/g2o_curve_fitting/cmake_modules)
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

#OpenCV路径
find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#PCL路径
find_package(PCL REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

#Ceres路径
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

#g2o路径
find_package( G2O REQUIRED )
include_directories(
        ${G2O_INCLUDE_DIRS}
        #        "/usr/include/eigen3"
)

find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

include_directories(ch3/useEigen)

#add_executable(Slam14 main.cpp ch3/useEigen/eigenMatrix.cpp ch3/useEigen/eigenMatrix.h)
#
#add_executable(useGeometry ch3/useGeometry/useGeometry.cpp ch3/useGeometry/useGeometry.h)
#
#add_executable(ch3_exercise ch3/ch3_exercise.cpp)
#
#
#add_executable(useSophus ch4/useSophus/useSophus.cpp)
#target_link_libraries(useSophus ${Sophus_LIBRARIES})
#
#
#
#
#add_executable(imageBasics ch5/imageBasics.cpp)
#target_link_libraries(imageBasics ${OpenCV_LIBS})
#
#
#
##set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/ch5)
#add_executable(joinMap ch5/joinMap.cpp)
#target_link_libraries(joinMap ${OpenCV_LIBS} ${PCL_LIBRARIES})
#
#
#
#add_executable(curve_fitting ch6/ceres_curve_fitting/main.cpp)
#target_link_libraries(curve_fitting ${OpenCV_LIBS} ${CERES_LIBRARIES})
#
#
#add_executable(g2o_curve_fitting ch6/g2o_curve_fitting/main.cpp)
#target_link_libraries(g2o_curve_fitting ${OpenCV_LIBS} g2o_core g2o_stuff)

#add_executable(feature_extraction ch7/feature/feature_extraction.cpp)
#target_link_libraries(feature_extraction ${OpenCV_LIBS})
#
#add_executable(pose_estimation ch7/feature/pose_estimation_2d2d.cpp)
#target_link_libraries(pose_estimation ${OpenCV_LIBS})

#add_executable(triangulation ch7/feature/triangulation.cpp ch7/feature/pose_estimation_2d2d.cpp )
#target_link_libraries(triangulation ${OpenCV_LIBS})


#add_executable(pose_estimation_3d2d ch7/feature/pose_estimation_3d2d.cpp ch7/feature/pose_estimation_2d2d.cpp)
#target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension ${CSPARSE_LIBRARY})

# add_executable(pose_estimation_3d3d ch7/feature/pose_estimation_3d3d.cpp ch7/feature/pose_estimation_3d2d.cpp ch7/feature/pose_estimation_2d2d.cpp)
# target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBS} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension ${CSPARSE_LIBRARY})

# add_executable(lkflow ch8/LKFlow/useLK.cpp)
# target_link_libraries(lkflow ${OpenCV_LIBS})

#稀疏直接法估计位姿
# add_executable(direct_sparse ch8/direct/direct_sparse.cpp)
# target_link_libraries(direct_sparse ${OpenCV_LIBS} g2o_core g2o_stuff g2o_solver_csparse g2o_types_sba g2o_csparse_extension ${CSPARSE_LIBRARY})

#半稠密直接法估计位姿
# add_executable(direct_semidense ch8/direct/direct_semidense.cpp)
# target_link_libraries(direct_semidense ${OpenCV_LIBS} g2o_core g2o_stuff g2o_solver_csparse g2o_types_sba g2o_csparse_extension ${CSPARSE_LIBRARY})

# add_executable(p181 ch7/feature/exercise/p181.cpp)
# target_link_libraries(p181 ${OpenCV_LIBS})

# add_executable(pose_estimation_3d2d ch7/feature/exercise/pose_estimation_3d2d.cpp ch7/feature/pose_estimation_2d2d.cpp)
# target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension ${CSPARSE_LIBRARY})
