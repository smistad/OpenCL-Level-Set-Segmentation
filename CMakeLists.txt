cmake_minimum_required(VERSION 2.8)
project(Level-Set-Function)

add_subdirectory(SIPL)
include_directories(${SIPL_INCLUDE_DIRS})

add_subdirectory(OpenCLUtilities)
include_directories(${OCL-Utilities_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DKERNELS_DIR=\\\"${PROJECT_SOURCE_DIR}/\\\"")

add_executable(levelSetSeg main.cpp)
target_link_libraries(levelSetSeg SIPL OCL-Utilities)