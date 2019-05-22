find_package(PCL REQUIRED COMPONENTS common io filters visualization)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(rgbd src/pointcloud_mapping.cpp)
target_link_libraries(rgbd ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

install(TARGETS rgbd 
        RUNTIME DESTINATION dense_RGBD/)     