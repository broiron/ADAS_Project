cmake_minimum_required(VERSION 3.0)
add_library(yoloDetect STATIC yoloDetect.cpp)
add_library(laneDetect STATIC laneDetect.cpp)
target_include_directories(yoloDetect PUBLIC ${CMAKE_SOURCE_DIR}/includes)
target_include_directories(laneDetect PUBLIC ${CMAKE_SOURCE_DIR}/includes)

find_package(OpenCV REQUIRED)
add_compile_options("-g")

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(yoloDetect ${OpenCV_LIBS})
target_link_libraries(laneDetect ${OpenCV_LIBS})
