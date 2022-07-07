cmake_minimum_required(VERSION 3.0)
project(final)

add_subdirectory(lib)

find_package(OpenCV REQUIRED)
add_compile_options("-g")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} yoloDetect)
target_link_libraries(${PROJECT_NAME} laneDetect)
