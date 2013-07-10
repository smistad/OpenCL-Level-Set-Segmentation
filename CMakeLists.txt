cmake_minimum_required(VERSION 2.8)
project(Level-Set-Function)

add_subdirectory(SIPL)
include_directories(${SIPL_INCLUDE_DIRS})

add_subdirectory(OpenCLUtilities)
include_directories(${OCL-Utilities_INCLUDE_DIRS})

find_package(OpenMP)
if(OPENMP_FOUND)
    message("-- OpenMP was detected. Using OpenMP to speed up some calculations.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -O3 -g -DKERNELS_DIR=\"${PROJECT_SOURCE_DIR}/\"")

add_executable(levelSetSeg main.cpp)
target_link_libraries(levelSetSeg SIPL OCL-Utilities)
