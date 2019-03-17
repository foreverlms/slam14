cmake_minimum_required(VERSION 2.8)
project(Slam14)

set(CMAKE_CXX_STANDARD 14)
#指定Sophus package路径
set(Sophus_DIR ch4/useSophus/Sophus/build)

include_directories(ch3/useEigen)

add_executable(Slam14 main.cpp ch3/useEigen/eigenMatrix.cpp ch3/useEigen/eigenMatrix.h)

add_executable(useGeometry ch3/useGeometry/useGeometry.cpp ch3/useGeometry/useGeometry.h)

add_executable(ch3_exercise ch3/ch3_exercise.cpp)

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(useSophus ch4/useSophus/useSophus.cpp)
target_link_libraries(useSophus ${Sophus_LIBRARIES})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(imageBasics ch5/imageBasics.cpp)
target_link_libraries(imageBasics ${OpenCV_LIBS})

find_package(PCL REQUIRED COMPONENT common io)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/ch5)
add_executable(joinMap ch5/joinMap.cpp)
target_link_libraries(joinMap ${OpenCV_LIBS} ${PCL_LIBRARIES})