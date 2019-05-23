find_package(PCL REQUIRED COMPONENTS common io filters visualization)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})

add_executable(rgbd src/pointcloud_mapping.cpp)
target_link_libraries(rgbd ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

add_executable(octomapping src/octomap_mapping.cpp)
target_link_libraries(octomapping ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})
install(TARGETS rgbd 
        RUNTIME DESTINATION dense_RGBD/)