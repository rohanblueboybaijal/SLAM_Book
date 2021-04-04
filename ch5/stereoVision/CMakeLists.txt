cmake_minimum_required(VERSION 2.8.2)
project(stereoVision)
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})