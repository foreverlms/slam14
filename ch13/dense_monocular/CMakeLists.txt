cmake_minimum_required(VERSION 2.8)

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

include_directories(include/)


# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/dense_monocular)
add_executable(dense_mapping src/dense_mapping.cpp)
target_link_libraries(dense_mapping ${Sophus_LIBRARIES} ${OpenCV_LIBRARIES})

install(TARGETS dense_mapping 
        RUNTIME DESTINATION dense_monocular/)