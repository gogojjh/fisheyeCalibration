project(fisheye_calibration)
cmake_minimum_required(VERSION 2.8)

set( CMAKE_BUILD_TYPE "Debug" )

find_package( OpenCV 3 REQUIRED)

include_directories( ${OPENCV_INCLUDE_DIR} )

add_executable( ${PROJECT_NAME} src/main.cpp)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
