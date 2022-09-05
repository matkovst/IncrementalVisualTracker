cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ivt)

# OpenCV
if (NOT OpenCV_DIR AND DEFINED ENV{OpenCV_DIR})
    set(OpenCV_DIR $ENV{OpenCV_DIR})
endif()
if(NOT OpenCV_DIR)
    message(FATAL_ERROR "You must specify OpenCV_DIR")
endif()
find_package(OpenCV REQUIRED HINTS ${OpenCV_DIR} core imgproc videoio highgui)

# Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

file(GLOB_RECURSE HEADERS src/*.h*)
file(GLOB_RECURSE SOURCES src/*.cpp)

add_executable(${PROJECT_NAME} main.cpp ${HEADERS} ${SOURCES})

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    )

target_link_libraries(
    ${PROJECT_NAME}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    )

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)