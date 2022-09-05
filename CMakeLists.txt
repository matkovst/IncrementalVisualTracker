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

MACRO(add_program NAME MAIN_FILE DOUBLE_PRECISION)
    add_executable(
        ${NAME} ${MAIN_FILE} ${HEADERS} ${SOURCES})

    target_compile_definitions(${NAME} PUBLIC DOUBLE_PRECISION=${DOUBLE_PRECISION})

    target_include_directories(
        ${NAME}
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        )
    
    target_link_libraries(
        ${NAME}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        Eigen3::Eigen
        )
        
    set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 14)

    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "Install prefix" FORCE)
    set(INSTALL_BIN_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH "Path where exe and dll will be installed")
    install(
        TARGETS ${NAME}
        CONFIGURATIONS Release
        RUNTIME DESTINATION bin
        )
ENDMACRO()

add_program(ivt_fast main.cpp 0)
add_program(ivt_accurate main.cpp 1)